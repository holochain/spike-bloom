use rand::prelude::*;
use std::collections::HashSet;
use std::sync::Arc;

type Hash = Arc<[u8; 32]>;
type Map = HashSet<Hash>;
type Node = Vec<Map>;
type Network = Vec<Node>;

fn rand_hash() -> Hash {
    let mut rng = rand::thread_rng();
    let mut out = [0; 32];
    rng.fill(&mut out[..]);
    Arc::new(out)
}

fn gen_hash(d: &[u8]) -> Hash {
    let mut out = [0; 32];
    let hash = blake2b_simd::Params::new()
        .hash_length(32)
        .to_state()
        .update(d)
        .finalize();
    out.copy_from_slice(hash.as_bytes());
    Arc::new(out)
}

fn gen_map(data_count: usize) -> Map {
    let mut out = HashSet::new();
    for _ in 0..data_count {
        let h = rand_hash();
        out.insert(h);
    }
    out
}

fn gen_node(data_count: usize, net_fact: usize) -> Node {
    let mut out = Vec::new();
    for _ in 0..net_fact {
        out.push(gen_map(data_count));
    }
    out
}

fn gen_network(data_count: usize, net_fact: usize) -> Network {
    let mut out = Vec::new();
    for _ in 0..net_fact {
        out.push(gen_node(data_count, net_fact));
    }
    out
}

fn is_node_consistent(node: &Node) -> bool {
    let first_map = node.get(0).unwrap();
    for map in node.iter() {
        if first_map != map {
            return false;
        }
    }
    true
}

fn is_network_consistent(network: &Network) -> bool {
    let first_map = network.get(0).unwrap().get(0).unwrap();
    for node in network.iter() {
        for map in node.iter() {
            if first_map != map {
                return false;
            }
        }
    }
    true
}

fn shuffle_network(network: &mut Network) {
    let mut rng = rand::thread_rng();
    for node in network.iter_mut() {
        node.shuffle(&mut rng);
    }
    network.shuffle(&mut rng);
}

fn sync_node(node: &mut Node) {
    let mut uber_set = HashSet::new();
    for map in node.iter_mut() {
        for h in map.drain() {
            uber_set.insert(h);
        }
    }
    for map in node.iter_mut() {
        *map = uber_set.clone();
    }
}

fn sync_network(network: &mut Network) {
    // sync the individual nodes
    for node in network.iter_mut() {
        sync_node(node);
    }
}

fn gen_bloom_for_map(map: &Map) -> bloomfilter::Bloom<Hash> {
    // 1 in 100 false positives...
    // we can get 1 in 1000 for ~2x the filter size, but may not be worth it
    // 1 in 100 pretty much guarantees full sync after two communications.
    const TGT_FP: f64 = 0.01;

    let len = map.len();

    let mut bloom = bloomfilter::Bloom::new_for_fp_rate(len, TGT_FP);

    for h in map.iter() {
        bloom.set(h);
    }

    bloom
}

type BytesTransferred = usize;

fn bloom_filter_sync_two_maps(map1: &mut Map, map2: &mut Map) -> BytesTransferred {
    const BLOOM_OVERHEAD: BytesTransferred = 0
        + 8 // bitmap bits
        + 4 // k_num
        + (8 * 4) // sip_keys
        ;

    let mut byte_tx = 0;

    let bloom1 = gen_bloom_for_map(map1);
    let bloom2 = gen_bloom_for_map(map2);

    for h in map1.iter() {
        if !bloom2.check(h) {
            byte_tx += h.len();
            map2.insert(h.clone());
        }
    }

    for h in map2.iter() {
        if !bloom1.check(h) {
            byte_tx += h.len();
            map1.insert(h.clone());
        }
    }

    byte_tx += BLOOM_OVERHEAD + bloom1.bitmap().len();
    byte_tx += BLOOM_OVERHEAD + bloom2.bitmap().len();

    byte_tx
}

fn bloom_filter_sync_first_map_to_others(network: &mut Network) -> BytesTransferred {
    let mut byte_tx = 0;

    let mut first_node = network.remove(0);
    {
        let first_map = first_node.get_mut(0).unwrap();

        for node in network.iter_mut() {
            byte_tx += bloom_filter_sync_two_maps(node.get_mut(0).unwrap(), first_map);
        }
    }

    network.push(first_node);

    byte_tx
}

fn hash_of_hashes<'a, I: IntoIterator<Item = &'a Hash>>(hashes: I) -> Hash {
    let mut uber_hash = Vec::new();
    for hash in hashes.into_iter() {
        uber_hash.extend_from_slice(&hash[..]);
    }
    gen_hash(&uber_hash)
}

fn rehash_filter_sync_two_maps(map1: &mut Map, map2: &mut Map) -> BytesTransferred {
    let mut byte_tx = 0;
    let hash1 = hash_of_hashes(map1.iter());
    let hash2 = hash_of_hashes(map2.iter());
    byte_tx += 32 + 32;

    if hash1 != hash2 {
        // node 1 sends all hashes
        byte_tx += map1.len() * 32;

        // node 2 requests ops it doesn't have from node 1
        for h in map1.iter() {
            if !map2.contains(h) {
                byte_tx += h.len();
                map2.insert(h.clone());
            }
        }

        // node 2 forwards ops it has that node 1 doesn't
        for h in map2.iter() {
            if !map1.contains(h) {
                byte_tx += h.len();
                map1.insert(h.clone());
            }
        }
    }

    byte_tx
}

fn rehash_filter_sync_first_map_to_others(network: &mut Network) -> BytesTransferred {
    let mut byte_tx = 0;
    let mut first_node = network.remove(0);
    {
        let first_map = first_node.get_mut(0).unwrap();

        for node in network.iter_mut() {
            byte_tx += rehash_filter_sync_two_maps(node.get_mut(0).unwrap(), first_map);
        }
    }

    network.push(first_node);

    byte_tx
}

type IterationCount = usize;
type SyncTime = std::time::Duration;

fn test_run(
    data_count: usize,
    net_fact: usize,
    net_sync_fn: fn(&mut Network) -> BytesTransferred,
) -> (IterationCount, BytesTransferred, SyncTime) {
    // generate a random network
    let mut network = gen_network(data_count, net_fact);

    // make sure the network is not consistent
    assert!(!is_network_consistent(&network));

    // sync the individual nodes
    for node in network.iter_mut() {
        // they start out inconsistent
        assert!(!is_node_consistent(node));

        // make them consistent
        sync_node(node);

        // make sure they are now consistent
        assert!(is_node_consistent(node));
    }

    // make sure the network as a whole is still inconsistent
    assert!(!is_network_consistent(&network));

    let start = std::time::Instant::now();
    let mut byte_tx = 0;
    let mut count = 0;
    loop {
        count += 1;

        // randomize which nodes speak to which nodes
        shuffle_network(&mut network);

        // run our inter-node syncro code
        byte_tx += net_sync_fn(&mut network);

        // sync the maps in individual nodes
        sync_network(&mut network);

        // check for consistency
        if is_network_consistent(&network) {
            break;
        }
    }

    (count, byte_tx, start.elapsed())
}

fn test_suite(
    name: &'static str,
    data_count: usize,
    net_fact: usize,
    net_sync_fn: fn(&mut Network) -> BytesTransferred,
) {
    println!(
        "running with {} ops / {}x{} nodes",
        data_count, net_fact, net_fact
    );
    use std::io::Write;
    let mut stdout = std::io::stdout();

    write!(stdout, "{} warmup ", name).unwrap();
    stdout.flush().unwrap();
    for _ in 1..=3 {
        write!(stdout, ".").unwrap();
        stdout.flush().unwrap();
        test_run(data_count, net_fact, net_sync_fn);
    }
    let mut it_count = Vec::new();
    let mut byte_tx = Vec::new();
    let mut sync_time = Vec::new();

    write!(stdout, "{} test ", name).unwrap();
    stdout.flush().unwrap();
    for _ in 1..=20 {
        write!(stdout, ".").unwrap();
        stdout.flush().unwrap();
        let (it, bt, tt) = test_run(data_count, net_fact, net_sync_fn);
        it_count.push(it);
        byte_tx.push(bt as f64 / 1024.0 / 1024.0);
        sync_time.push(tt.as_secs_f64());
    }
    println!("done.");

    use stats::*;
    let it_count_dev = stddev(it_count.iter().cloned());
    let it_count = mean(it_count.iter().cloned());
    let byte_tx_dev = stddev(byte_tx.iter().cloned());
    let byte_tx = mean(byte_tx.iter().cloned());
    let sync_time_dev = stddev(sync_time.iter().cloned());
    let sync_time = mean(sync_time.iter().cloned());

    println!(
        "{} iterations: {:.01}±{:.04}, MiB tranferred: {:.04}±{:.04} in {:.04}±{:.04} s",
        name, it_count, it_count_dev, byte_tx, byte_tx_dev, sync_time, sync_time_dev,
    );
}

pub fn bloom_test_suite(data_count: usize, net_fact: usize) {
    test_suite(
        "bloom",
        data_count,
        net_fact,
        bloom_filter_sync_first_map_to_others,
    );
}

pub fn rehash_test_suite(data_count: usize, net_fact: usize) {
    test_suite(
        "rehash",
        data_count,
        net_fact,
        rehash_filter_sync_first_map_to_others,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        bloom_test_suite(20, 10);
        rehash_test_suite(20, 10);
    }
}

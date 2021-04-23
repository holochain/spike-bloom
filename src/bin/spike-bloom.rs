fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert!(args.len() == 3);
    let data_count: usize = args.get(1).unwrap().parse().unwrap();
    let net_fact: usize = args.get(2).unwrap().parse().unwrap();
    spike_bloom::bloom_test_suite(data_count, net_fact);
    spike_bloom::rehash_test_suite(data_count, net_fact);
}

#[derive(Debug)]
enum IpAddrKind {
    V4,
    V6,
}

enum IpAddr {
    V4(u8, u8, u8, u8),
    V6(String),
}

struct IpAddrWithKind {
    kind: IpAddrKind,
    address: String,
}

fn main() {
    let home = IpAddrWithKind {
        kind: IpAddrKind::V4,
        address: String::from("127.0.0.1"),
    };
    println!("kind: {:?}, address: {}", home.kind, home.address);

    let loopback = IpAddrWithKind {
        kind: IpAddrKind::V6,
        address: String::from("::1"),
    };
    println!("kind: {:?}, address: {}", loopback.kind, loopback.address);

    let home = IpAddr::V4(127, 0, 0, 1);
    println!("{}", home);
}

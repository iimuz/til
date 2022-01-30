fn main() {
    let user = build_user(String::from("hoge@example.com"), String::from("hoge"));
    print_user(&user);

    let user2 = User {
        email: String::from("geho@example.com"),
        username: String::from("geho"),
        ..user
    };
    print_user(&user2);
}

struct User {
    username: String,
    email: String,
    sign_in_count: u64,
    active: bool,
}

fn build_user(email: String, username: String) -> User {
    User {
        email,    // email: emailと同じ
        username, // username: useranmeと同じ
        active: true,
        sign_in_count: 1,
    }
}

fn print_user(user: &User) {
    println!("username: {}", user.username);
    println!("email: {}", user.email);
    println!("sign in count: {}", user.sign_in_count);
    println!("active: {}", user.active);
}

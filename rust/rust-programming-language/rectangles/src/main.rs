#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }

    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }

    fn square(size: u32) -> Rectangle {
        Rectangle { width: size, height: size}
    }
}

fn main() {
    let rect1 = Rectangle {
        width: 30,
        height: 20,
    };
    println!(
        "The area of the rectangle is {} suare pixels.",
        area(&rect1)
    ); // 関数を呼び出して計算
    println!(
        "The area of the rectangle is {} suare pixels.",
        rect1.area()
    ); // メソッドを呼び出して計算

    println!("rect1 is {:?}", rect1);
    println!("rect1 is {:#?}", rect1);

    let rect2 = Rectangle {
        width: 20,
        height: 10,
    };
    let rect3 = Rectangle {
        width: 60,
        height: 45,
    };
    println!("Can rect1 hold rect2? {}", rect1.can_hold(&rect2));
    println!("Can rect1 hold rect3? {}", rect1.can_hold(&rect3));

    let square = Rectangle::square(3);
    println!("square rect is {:?}", square);
}

fn area(rectangle: &Rectangle) -> u32 {
    rectangle.height * rectangle.width
}

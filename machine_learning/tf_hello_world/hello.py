import tensorflow as tf


def main() -> None:
    hello = tf.constant('hello world')
    s = tf.Session()
    print(s.run(hello))


if __name__ == '__main__':
    main()

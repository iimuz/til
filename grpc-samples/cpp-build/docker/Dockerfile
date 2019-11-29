FROM buildpack-deps:stretch-scm
LABEL maintainer iimuz

# set locale
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
    apt-utils \
    locales && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  echo en_US.UTF-8 UTF-8 > /etc/locale.gen && \
  locale-gen && \
  update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# gosu
ENV GOSU_VERSION=1.10
RUN set -x && \
  fetchDeps=' \
    ca-certificates \
    wget \
  ' && \
  apt update && \
  apt install -y --no-install-recommends $fetchDeps && \
  dpkgArch="$(dpkg --print-architecture | awk -F- '{ print $NF }')" && \
  wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$dpkgArch" && \
  chmod +x /usr/local/bin/gosu && \
  gosu nobody true && \
  apt clean && \
  rm -rf /var/lib/apt/lists/* && \
  apt autoremove -y

ENV USER_NAME=cppdev \
  HOME=/home/cppdev \
  USER_ID=1000 \
  GROUP_ID=1000
RUN set -x && \
  adduser ${USER_NAME} --uid ${USER_ID} --disabled-password --gecos ""

# setup tools
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
    cmake \
    make && \
  apt-get install -y \
    autoconf \
    build-essential \
    libtool \
    pkg-config && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# clang
RUN echo deb http://apt.llvm.org/stretch/ llvm-toolchain-stretch-5.0 main >> /etc/apt/sources.list && \
  echo deb-src http://apt.llvm.org/stretch/ llvm-toolchain-stretch-5.0 main >> /etc/apt/sources.list.d/deb-src.list  && \
  wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
    clang-5.0 \
    libclang-common-5.0-dev \
    libclang-5.0-dev \
    libclang1-5.0 \
    libclang1-5.0-dbg \
    libllvm5.0 \
    libllvm5.0-dbg \
    lldb-5.0 \
    llvm-5.0 \
    llvm-5.0-dev \
    llvm-5.0-runtime \
    clang-format-5.0 \
    liblldb-5.0-dev \
    lld-5.0 \
    libfuzzer-5.0-dev && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  cd /usr/bin && \
  ln -s ../lib/llvm-5.0/bin/clang clang && \
  ln -s ../lib/llvm-5.0/bin/clang++ clang++ && \
  ln -s ../lib/llvm-5.0/bin/clang-cpp clang-cpp && \
  ln -s ../lib/llvm-5.0/bin/clang-format clang-format && \
  ln -s ../lib/llvm-5.0/bin/git-clang-format git-clang-format && \
  ln -s ./clang-format-diff-5.0 clang-format-diff
ENV CC=clang \
  CXX=clang++

# grpc and protobuf
ENV GRPC_VER=v1.15.0
RUN set -x && \
  cd /opt && \
  : "get source code" && \
  git clone --depth=1 -b ${GRPC_VER} https://github.com/grpc/grpc.git && \
  cd grpc && \
  git submodule update --init && \
  : "install protobuf" && \
  cd third_party/protobuf && \
  ./autogen.sh && \
  ./configure CC=${CC} CXX=${CXX} && \
  make -j4 && \
  make install && \
  ldconfig && \
  : "install grpc" && \
  cd ../.. && \
  make -j4 && \
  make install && \
  ldconfig && \
  : "clean" && \
  cd .. && \
  rm -rf grpc


ENV SOURCE_DIR=/src
RUN set -x && mkdir ${SOURCE_DIR}

ADD ./entrypoint.sh /
RUN chmod +x /entrypoint.sh
WORKDIR ${HOME}
ENTRYPOINT ["/entrypoint.sh"]
CMD ["clang"]

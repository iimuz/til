package main

import (
	"fmt"
	"github.com/fabiokung/shm"
	"io"
	"os"
	"syscall"
	"time"
	"unsafe"
)

// command line tool info
type cl struct {
	OutStream io.Writer
	ErrStream io.Writer
}

// version info
var (
	version  string
	revision string
)

// exit status
const (
	ExitCodeOK int = iota
	ExitCodeError
)

// main entrypoint
func main() {
	instance := &cl{OutStream: os.Stdout, ErrStream: os.Stderr}
	os.Exit(instance.run(os.Args))
}

// run is client process
func (c *cl) run(args []string) int {
	fmt.Fprintf(c.OutStream, "start shared memory client using golang.\n")

	// 共有メモリのエントリポイントを取得(読み込み専用)
	name := string("shared_memory")
	file, err := shm.Open(name, os.O_RDWR, 0600)
	if err != nil {
		fmt.Fprintf(c.ErrStream, "open error: %v\n", err)
		return ExitCodeError
	}
	defer file.Close()
	defer shm.Unlink(file.Name())

	// 共有メモリを[]byteとして取得
	fd := int(uintptr(file.Fd()))
	buff, err := syscall.Mmap(fd, 0, 1024, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		fmt.Fprintf(c.ErrStream, "mmap error: %v\n", err)
		return ExitCodeError
	}

	// 共有メモリを[]intと仮定して読み込み
	const LOOP_NUM = 10
	b := (*[256]int)(unsafe.Pointer(&buff[0]))[:]
	for i := 0; i < LOOP_NUM; i++ {
		fmt.Fprintf(c.OutStream, "%v\n", b[0])
		time.Sleep(1 * time.Second)
	}

	fmt.Fprintf(c.OutStream, "end shared memory client using golang.\n")
	return ExitCodeOK
}

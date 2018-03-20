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

// main is entrypoint of this binary
func main() {
	instance := &cl{OutStream: os.Stdout, ErrStream: os.Stderr}
	os.Exit(instance.run(os.Args))
}

func (c *cl) run(args []string) int {
	fmt.Fprintf(c.OutStream, "start shared memory server using golang.\n")

	name := string("shmgo")
	file, err := shm.Open(name, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		fmt.Fprintf(c.ErrStream, "open error: %v\n", err)
		return ExitCodeError
	}
	defer file.Close()
	defer shm.Unlink(file.Name())

	if err := syscall.Ftruncate(int(file.Fd()), int64(1024)); err != nil {
		fmt.Fprintf(c.ErrStream, "truncate error: %v\n", err)
		return ExitCodeError
	}

	fd := int(uintptr(file.Fd()))
	buff, err := syscall.Mmap(fd, 0, 1024, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		fmt.Fprintf(c.ErrStream, "mmap error: %v\n", err)
		return ExitCodeError
	}

	const LOOP_NUM = 10
	b := (*[256]int)(unsafe.Pointer(&buff[0]))[:]
	for i := 0; i < LOOP_NUM; i++ {
		b[0] = i
		fmt.Fprintf(c.OutStream, "%v\n", b[0])
		time.Sleep(1 * time.Second)
	}

	fmt.Fprintf(c.OutStream, "end shared memory server using golang.\n")
	return ExitCodeOK
}

package main

import (
	"fmt"
	"github.com/fabiokung/shm"
	"io"
	"os"
	"syscall"
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

func (c *cl) run(args []string) int {
	fmt.Fprintf(c.OutStream, "start shared memory server using golang.\n")

	file, err := shm.Open("my_region", os.O_RDWR|os.O_CREATE, 0600)
	if err != nil {
		fmt.Fprintf(c.ErrStream, "shared memory open error: %v", err)
		return ExitCodeError
	}
	defer file.Close()
	defer shm.Unlink(file.Name())

	if err := syscall.Ftruncate(int(file.Fd()), 1024); err != nil {
		fmt.Fprintf(c.ErrStream, "shared memory truncate error: %v", err)
		return ExitCodeError
	}

	data := uint8(100)
	if err := file.Write(([]byte)(1)); err != nil {
		fmt.Fprintf(c.ErrStream, "shared memory write error: %v", err)
		return ExitCodeError
	}

	fmt.Fprintf(c.OutStream, "end shared memory server using golang.\n")
	return ExitCodeOK
}

package main

import (
	"fmt"
	"io"
	"os"
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
	fmt.Fprintf(c.OutStream, "start shared memory client using golang.\n")

	fmt.Fprintf(c.OutStream, "end shared memory client using golang.\n")
	return ExitCodeOK
}

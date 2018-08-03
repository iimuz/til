package main

import "github.com/labstack/echo"

func main() {
	e := echo.New()
	e.File("/", "web/app/index.html")
	e.Static("/static", "web/static")
	err := e.Start(":4000")
	if err != nil {
		panic(err)
	}
}

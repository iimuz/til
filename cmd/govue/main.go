package main

import (
	"github.com/iimuz/go-vue-example/pkg/controllers"
	"github.com/labstack/echo"
)

func main() {
	e := echo.New()
	tasks := controllers.TaskController{Router: e.Router()}
	tasks.Setup()
	e.File("/", "web/app/index.html")
	e.Static("/static", "web/static")
	err := e.Start(":4000")
	if err != nil {
		panic(err)
	}
}

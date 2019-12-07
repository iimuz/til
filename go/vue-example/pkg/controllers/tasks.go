package controllers

import "github.com/labstack/echo"

type TaskController struct {
	Router *echo.Router
}

type ResponseMap map[string]interface{}

func (controller *TaskController) listTasks(c echo.Context) error {
	body := ResponseMap{}
	body["success"] = true
	items := []ResponseMap{
		ResponseMap{
			"id":   0,
			"body": "test",
			"done": false,
		},
	}
	body["items"] = items
	body["count"] = 1
	return c.JSON(200, body)
}

/// Setup sets up routes for the task controller.
func (controller *TaskController) Setup() {
	controller.Router.Add("GET", "/tasks", controller.listTasks)
}

# route_manager.py
from flask import Flask

class RouteManager:
    def __init__(self):
        self.app = None

    def init_app(self, app: Flask):
        self.app = app

    def register_blueprints(self):
        """动态导入并注册各个模块的路由"""
        from api_routes.example_app import add_example_routes
        add_example_routes(self.app)

# 全局实例
route_manager = RouteManager()
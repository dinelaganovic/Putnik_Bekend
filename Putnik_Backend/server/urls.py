from django.contrib import admin
from django.urls import path
from app.views import handle_map_and_agent

urlpatterns = [
    path('admin/', admin.site.urls),
    path('game', handle_map_and_agent, name='handle-map-request'),
]

from django.urls import path

from .views import home,about,portfolio,delete_stock,risk_analysis

urlpatterns = [
    path('', home, name='home'),
    path('about/', about, name='about'),
    path('portfolio/', portfolio, name='portfolio'),
    path('risk_analysis/', risk_analysis, name='risk_analysis'),
    path('deletestock/<stock_symbol>', delete_stock, name='delete_stock'),
]

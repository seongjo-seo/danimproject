from django.shortcuts import render
from django.http.response import HttpResponse
# Create your views here.

# 데이터 더미로 던져주는 것.

def index(request):
    return HttpResponse("Hello ssj")
# custom_filters.py

from django import template

register = template.Library()

@register.filter(name='get_item')
def get_item(dictionary, key):
    return dictionary.get(key)

@register.filter(name='add_index')
def add_index(data, start=1):
    return enumerate(data, start)
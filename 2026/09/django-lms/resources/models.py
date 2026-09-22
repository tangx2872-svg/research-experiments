from django.db import models
from users.models import User
from courses.models import Course
from django.urls import reverse
from django.utils import timezone
import os
from django.conf import settings

# Create your models here.
class Resource(models.Model):
    resource_name = models.CharField(max_length=200, blank=False)
    resource_file = models.FileField(blank=False)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)

    def __str__(self):
        return self.resource_name

    def get_absolute_url(self):
        return reverse('courses:list')

from django.db.models.signals import post_delete
from django.dispatch import receiver
from django_lms.files import remove_unreferenced_file


@receiver(post_delete, sender=Resource)
def cleanup_attachment(sender, instance, **kwargs):
    field = instance.resource_file
    remove_unreferenced_file(field.storage, field.name)

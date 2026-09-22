from django.shortcuts import render
import datetime
from django.contrib.auth.mixins import (LoginRequiredMixin,
                                        PermissionRequiredMixin)
from django.urls import reverse
from django.contrib import messages
from django.views import generic
from django.shortcuts import get_object_or_404
from users.models import User
from courses.models import Course, Enrollment
from courses.forms import CreateCourseForm
from assignments.models import Assignment
from resources.models import Resource

# Create your views here.
class CreateCourse(LoginRequiredMixin, generic.CreateView):
    form_class = CreateCourseForm
    model = Course

    def get(self, request,*args, **kwargs):
        self.object = None
        context_dict = self.get_context_data()
        context_dict.update(user_type=self.request.user.user_type)
        return self.render_to_response(context_dict)
    
    def form_valid(self, form):
        form.instance.teacher = self.request.user
        return super(CreateCourse, self).form_valid(form)
    
class CourseDetail(generic.DetailView):
    model = Course

    def get_context_data(self,**kwargs):
        assignments = Assignment.objects.filter(course=self.kwargs['pk'])
        resources = Resource.objects.filter(course=self.kwargs['pk'])
        context = super(CourseDetail, self).get_context_data(**kwargs)
        context['assignments'] = assignments
        context['resources'] = resources
        return context

class ListCourse(generic.ListView):
    model = Course

class EnrollCourse(LoginRequiredMixin, generic.RedirectView):

    def get_redirect_url(self, *args, **kwargs):
        return reverse('courses:detail', kwargs={'pk':self.kwargs.get('pk')})
    
    def get(self, *args, **kwargs):
        course = get_object_or_404(Course, pk=self.kwargs.get('pk'))

        try:
            Enrollment.objects.create(student=self.request.user, course=course)
        except:
            messages.warning(self.request, '您已加入这门课程，无需重复加入。')
        else:
            messages.success(self.request, '已成功加入课程。')
        return super().get(self.request, *args, **kwargs)

class UnenrollCourse(LoginRequiredMixin, generic.RedirectView):

    def get_redirect_url(self, *args, **kwargs):
        return reverse('courses:detail', kwargs={'pk':self.kwargs.get('pk')})

    def get(self, *args, **kwargs):

        try:
            enrollment = Enrollment.objects.filter(
                student=self.request.user,
                course__pk=self.kwargs.get('pk')
            ).get()
        except Enrollment.DoesNotExist:
            messages.warning(self.request, '您尚未加入这门课程。')
        else:
            enrollment.delete()
            messages.success(self.request, '已退出课程。')
        return super().get(self.request, *args, **kwargs)

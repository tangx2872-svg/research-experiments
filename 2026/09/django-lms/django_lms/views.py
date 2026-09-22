from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.exceptions import PermissionDenied
from django.shortcuts import render
from django.views import generic
from courses.models import Course
from assignments.models import SubmitAssignment


def index(request):
    return render(request, 'index.html')


class UserProfile(LoginRequiredMixin, generic.ListView):
    model = Course
    template_name = 'user_profile.html'

    def get_queryset(self):
        if str(self.request.user.pk) != str(self.kwargs['pk']):
            raise PermissionDenied('只能查看自己的个人页。')
        if self.request.user.user_type == 2:
            return Course.objects.filter(teacher=self.request.user)
        return Course.objects.filter(students=self.request.user)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.request.user.user_type == 1:
            context['my_submissions'] = SubmitAssignment.objects.filter(author=self.request.user).select_related('assignment_ques__course').order_by('-updated_at', '-pk')
        return context

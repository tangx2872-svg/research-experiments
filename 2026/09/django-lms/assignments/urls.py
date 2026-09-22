from django.conf.urls import url
from assignments import views
from assignments import centers
from django.conf import settings
from django.conf.urls.static import static

app_name = 'assignments'
urlpatterns = [
    url(r'^center/$', centers.assignment_center, name='center'),
    url(r'^feedback/$', centers.feedback_center, name='feedback'),
    url(r'^submission/download/(?P<pk>[0-9]+)/$', views.download_submission, name='download'),
    url(r'^create/$', views.CreateAssignment.as_view(), name="create"),
    url(r'^detail/(?P<pk>[-\w]+)/$', views.AssignmentDetail.as_view(), name='detail'),
    url(r'^update/(?P<pk>[-\w]+)/$', views.UpdateAssignment.as_view(), name='update'),
    url(r'^delete/(?P<pk>[-\w]+)/$', views.DeleteAssignment.as_view(), name="delete"),
    url(r'^submit/(?P<pk>[0-9]+)/$', views.submit_assignment, name='submit'),
    url(r'^submission/detail/(?P<pk>[-\w]+)/$', views.SubmitAssignmentDetail.as_view(), name="submit_detail"),
    url(r'^submission/delete/(?P<pk>[-\w]+)/$', views.delete_view, name="submit_delete"),
    url(r'^grade/(?P<pk>[-\w]+)/$', views.grade_assignment, name='grade')
]

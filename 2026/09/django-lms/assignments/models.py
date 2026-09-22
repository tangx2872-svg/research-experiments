from django.db import models
from users.models import User
from courses.models import Course
from django.urls import reverse
from django.utils import timezone
from django.core.validators import MaxValueValidator, MinValueValidator

# Create your models here.
class Assignment(models.Model):
    assignment_name = models.CharField(max_length=200, blank=False)
    assignment_description = models.TextField(blank=False)
    start_date = models.DateTimeField(default=timezone.now)
    due_date = models.DateTimeField()
    course = models.ForeignKey(Course, on_delete=models.CASCADE)

    @property
    def is_open(self):
        now = timezone.now()
        return not self.course.is_archived and self.start_date <= now < self.due_date

    @property
    def status_label(self):
        if self.course.is_archived:
            return '课程已归档'
        if timezone.now() < self.start_date:
            return '未开始'
        return '进行中' if self.is_open else '已截止'

    def __str__(self):
        return self.assignment_name

    def get_absolute_url(self):
        return reverse('assignments:detail', kwargs={'pk': self.pk})

class SubmitAssignment(models.Model):
    author = models.ForeignKey(User, related_name='assignment', on_delete=models.CASCADE)
    topic = models.CharField(max_length=200, blank=False)
    description = models.TextField(blank=False)
    assignment_file = models.FileField(blank=False, upload_to='assignments')
    submitted_date = models.DateTimeField(default=timezone.now)
    assignment_ques = models.ForeignKey(Assignment, related_name="question", on_delete=models.CASCADE, null=True)
    # NULL marks a retained historical version; slot 1 is the effective submission.
    current_slot = models.PositiveSmallIntegerField(default=1, null=True, editable=False)
    updated_at = models.DateTimeField(default=timezone.now)
    graded = models.BooleanField(default=False)
    feedback = models.TextField(blank=True)
    graded_at = models.DateTimeField(null=True, blank=True)
    grade = models.IntegerField(
        default=0,
        validators=[
            MaxValueValidator(100),
            MinValueValidator(0)
        ]
    )

    def __str__(self):
        return self.topic

    def grade_assignment(self, grade, feedback='', grader=None):
        if self.graded and not self.grade_records.exists():
            GradeRecord.objects.create(submission=self, grade=self.grade, feedback=self.feedback, created_at=self.graded_at)
        self.grade = grade
        self.graded = True
        self.feedback = feedback
        self.graded_at = timezone.now()
        self.save()
        if grader is not None:
            GradeRecord.objects.create(submission=self, grader=grader, grade=grade, feedback=feedback)

    def get_absolute_url(self):
        return reverse('assignments:submit_detail', kwargs={'pk': self.pk})

    class Meta:
        constraints = [models.UniqueConstraint(fields=['author', 'assignment_ques', 'current_slot'], name='unique_current_submission')]


from django.db.models.signals import post_delete
from django.dispatch import receiver
from django_lms.files import remove_unreferenced_file


@receiver(post_delete, sender=SubmitAssignment)
def cleanup_attachment(sender, instance, **kwargs):
    field = instance.assignment_file
    remove_unreferenced_file(field.storage, field.name)


class GradeRecord(models.Model):
    submission = models.ForeignKey(SubmitAssignment, related_name='grade_records', on_delete=models.CASCADE)
    grader = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
    grade = models.PositiveSmallIntegerField(validators=[MaxValueValidator(100)])
    feedback = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now, null=True, blank=True)

    class Meta:
        ordering = ['-created_at', '-pk']

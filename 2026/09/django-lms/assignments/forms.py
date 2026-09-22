from django import forms
from django.utils import timezone
from assignments.models import SubmitAssignment, Assignment
from courses.models import Course
from django_lms.files import validate_upload


class GradeAssignmentForm(forms.ModelForm):
    class Meta:
        model = SubmitAssignment
        fields = ['grade', 'feedback']
        labels = {'grade': '成绩', 'feedback': '批改评语'}
        help_texts = {'grade': '请输入 0 至 100 之间的整数。'}


class CreateAssignmentForm(forms.ModelForm):
    class Meta:
        model = Assignment
        fields = ('assignment_name', 'assignment_description', 'due_date', 'course')
        labels = {'assignment_name': '作业名称', 'assignment_description': '作业要求', 'due_date': '截止时间（北京时间）', 'course': '所属课程'}
        widgets = {'due_date': forms.DateTimeInput(format='%Y-%m-%dT%H:%M', attrs={'type': 'datetime-local'})}

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user')
        super().__init__(*args, **kwargs)
        self.fields['course'].queryset = Course.objects.filter(teacher=user, is_archived=False)
        if self.instance.pk:
            self.fields['course'].disabled = True

    def clean_due_date(self):
        value = self.cleaned_data['due_date']
        if value <= timezone.now():
            if not self.instance.pk or value.replace(second=0, microsecond=0) != self.instance.due_date.replace(second=0, microsecond=0):
                raise forms.ValidationError('新的截止时间必须晚于当前时间。')
        if value <= self.instance.start_date:
            raise forms.ValidationError('截止时间必须晚于发布时间。')
        return value


class SubmitAssignmentForm(forms.ModelForm):
    class Meta:
        model = SubmitAssignment
        fields = ('topic', 'description', 'assignment_file')
        labels = {'topic': '提交标题', 'description': '作业说明', 'assignment_file': '作业文件'}
        help_texts = {'assignment_file': '最大 20 MB；支持常见文档、图片、ZIP、.py、.ipynb。更新时不选择文件即可保留原附件。'}
        widgets = {'assignment_file': forms.FileInput()}

    def clean_assignment_file(self):
        file = self.cleaned_data['assignment_file']
        if 'assignment_file' in self.files:
            validate_upload(file)
        return file

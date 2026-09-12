from pydantic import BaseModel, EmailStr, Field


class Student(BaseModel):

    name: str = 'Niteesh'
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing the grade of a student')

new_student = {'email': 'abc@gmail.com', 'cgpa': 5.0}

student = Student(**new_student)

print(student)
print(type(student))
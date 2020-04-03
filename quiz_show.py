
class Quiz:

    def __init__(self,text):
        self.text = text
        self.yes = None
        self.no = None


class QuizManager:

    def __init__(self):
        q1 = Quiz("SNS 사진 올린다?")
        q2 = Quiz("먹는걸 좋아한다. Q1 YES")
        q3 = Quiz("힙하다는 얘기를 자주 듣는다 Q1 No")
        q4 = Quiz("Q1 YES YES")
        q5 = Quiz("Q1 YES YES YES")

        q1.yes = q2
        q1.no = q3
        q2.yes = q4
        q4.yes = q5

        self.start_quiz = q1

    def getFirstQuiz(self):
        return self.start_quiz

class QuizUI:

    def __init__(self):
        manager = QuizManager()
        self.playShow(manager.getFirstQuiz())

    def playShow(self, quiz):

        if quiz == None:
            print("THE END")
            return

        answer = input(quiz.text)

        if answer == 'y':
            self.playShow(quiz.yes)
        elif answer == 'n':
            self.playShow(quiz.no)


ui = QuizUI()
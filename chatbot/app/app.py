import random

from telegram import Telegram
from chain import Chain


class MainApp:
    def __init__(self):
        self.telegram = Telegram(self.handle_message)
        self.chain = Chain()
        self.intro_templates = [
            "I am your friendly menstrual health assistant. Here to provide guidance and support on topics like:",
            "Hello! I’m here to help you with all things related to menstrual health. Feel free to ask about:",
            "Welcome to your menstrual health companion! I can provide information and tips about:",
            "Hi there! I’m your guide to menstrual well-being. Let’s talk about topics such as:",
            "Greetings! Your menstrual health matters, and I’m here to assist you with queries related to:"
        ]

        self.topics = [
            "- Understanding your menstrual cycle",
            "- Common period symptoms",
            "- Managing menstrual pain",
            "- Ovulation and fertility",
            "- Tips for a healthy cycle",
            "- Period tracking and prediction",
            "- Diet and exercise during periods",
            "- Premenstrual syndrome (PMS)",
            "- Menstrual hygiene practices",
            "- Irregular periods",
            "- When to consult a doctor",
            "- Contraception and periods",
            "- Period myths and facts",
            "- Hormonal changes during menstruation"
        ]

        self.closing_text = "\n\nFeel free to ask any Menstrual period related inquiries you have! I'm here to help!"

    def get_intro_response(self):
        random.shuffle(self.topics)
        intro = random.choice(self.intro_templates)
        topics_list = "\n".join(self.topics[:5])
        response = f"{intro}\n{topics_list}{self.closing_text}"
        return response

    def handle_message(self, message, chat_id):
        # print(f"Received message: {message}")
        if message.lower() in ['/start', 'hello', 'hi', 'greetings']:
            response = self.get_intro_response()
        else:
            response = self.chain.get_response(message, chat_id)
        # print(f"Generated response: {response}")

        self.telegram.send_message(chat_id, response)

    def run(self):
        print("Starting the bot...")
        self.telegram.start()


if __name__ == '__main__':
    app = MainApp()
    app.run()

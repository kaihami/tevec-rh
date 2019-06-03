import json

class ForecastService():
    """
    ... Develop our code ...
    """

    def __init__(self, model):
        """
        ... Just an example ...
        """
        self.model = model
        super().__init__(self)

    def apply(self, request):
        """
        ... Develop our code ...
        """
        return self


if __name__ == "__main__":

    api = ForecastService()

    while True:
        user_input = input("Digite o input digite")
        user_input = json.loads(user_input)
        output = api.apply(json_input)
        print(output)
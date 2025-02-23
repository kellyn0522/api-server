import numpy as np

class AndModel:
    def __init__(self):
        # 파라메터
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)

    def train(self):
        learning_rate = 0.1
        epochs = 20
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([0, 0, 0, 1])        
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # 총 입력 계산
                total_input = np.dot(inputs[i], self.weights) + self.bias
                # 예측 출력 계산
                prediction = self.step_function(total_input)
                # 오차 계산
                error = outputs[i] - prediction
                print(f'inputs[i] : {inputs[i]}')
                print(f'weights : {self.weights}')
                print(f'bias before update: {self.bias}')
                print(f'prediction: {prediction}')
                print(f'error: {error}')
                # 가중치와 편향 업데이트
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error
                print('====')        

    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, input_data):
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)    



class OrModel:
    def __init__(self):
        # 파라메터
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)

    def train(self):
        learning_rate = 0.1
        epochs = 20
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([0, 1, 1, 1])        
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # 총 입력 계산
                total_input = np.dot(inputs[i], self.weights) + self.bias
                # 예측 출력 계산
                prediction = self.step_function(total_input)
                # 오차 계산
                error = outputs[i] - prediction
                print(f'inputs[i] : {inputs[i]}')
                print(f'weights : {self.weights}')
                print(f'bias before update: {self.bias}')
                print(f'prediction: {prediction}')
                print(f'error: {error}')
                # 가중치와 편향 업데이트
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error
                print('====')        

    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, input_data):
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)   



class NotModel:
    def __init__(self):
        # 파라메터 : not 연산은 가중치, 편향 필요 없음 
        pass

    def train(self):
        print("NOT 연산은 학습 필요 없음.")

    def step_function(self, x):
        return 1 if x == 0 else 0  # 기준을 반대로 설정 
    
    def predict(self, input_data):
        return self.step_function(input_data)   



from sklearn.neural_network import MLPClassifier

class XorModel:
    def __init__(self):
        # xor은 선형 분리가 불가능 (단순 퍼셉트론 X) -> 다층 퍼셉트론 MLP 사용 
        # max_iter=1000으로 하였지만, 모델 학습이 제대로 이루어지지 않아서 5000으로 변경 
        self.model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=5000, solver='adam', random_state=42)

    def train(self):
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        # 입력값 정규화 부족으로 MLPClassifier의 경우 0과 1의 학습이 어려움 -> 입력값을 변환하여 학습 진행 
        inputs = inputs * 2 - 1     # 0 -> -1, 1 -> 1
        outputs = np.array([0, 1, 1, 0]) 
        # 모델 학습        
        self.model.fit(inputs, outputs)

    def predict(self, input_data):
        input_array = np.array(input_data).reshape(1, -1) * 2 - 1   # 0 -> -1, 1 -> 1
        return int(self.model.predict(input_array)[0])
    

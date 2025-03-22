import numpy as np
import time

# 복잡한 다변수 함수 정의
def func(x, y):
    return np.sin(x) * np.exp(y) + x**3 * y**2

# 수치미분 계산
def numerical_gradient(x, y):
    epsilon = 1e-4
    df_dx = (func(x + epsilon, y) - func(x - epsilon, y)) / (2 * epsilon)
    df_dy = (func(x, y + epsilon) - func(x, y - epsilon)) / (2 * epsilon)
    return df_dx, df_dy

# 오차역전파 계산
def backprop_gradient(x, y):
    df_dx = np.cos(x) * np.exp(y) + 3 * x**2 * y**2
    df_dy = np.sin(x) * np.exp(y) + 2 * x**3 * y
    return df_dx, df_dy

# 반복 횟수 설정(1번만 실행하면 둘다 0second로 뜨기 때문에 횟수를 늘림으로써 차이를 보기 쉽게 함)
num_iterations = 100000
x, y = 3.0, 4.0

# 수치미분 시간 측정
start_time = time.time()
for _ in range(num_iterations):
    num_grad = numerical_gradient(x, y)
numerical_time = time.time() - start_time

# 역전파 시간 측정
start_time = time.time()
for _ in range(num_iterations):
    back_grad = backprop_gradient(x, y)
backprop_time = time.time() - start_time

# 결과 출력
print(f"After {num_iterations} iterations:")
print(f"Numerical Gradient Time: {numerical_time:.6f} seconds")
print(f"Backpropagation Gradient Time: {backprop_time:.6f} seconds")

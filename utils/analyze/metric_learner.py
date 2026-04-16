import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from sklearn.model_selection import train_test_split
import cvxpy as cp
import torch
import torch.optim as optim

class ColorMetricLearner:
    """
    Обучение линейного преобразования, сохраняющего расстояния между цветами
    """
    
    def __init__(self, input_dim, output_dim=3, method='gradient', device='cpu'):
        """
        Parameters:
        -----------
        input_dim : int
            Размерность входных эмбеддингов
        output_dim : int
            Размерность выходного пространства (обычно 3 для цветов)
        method : str
            Метод оптимизации: 'gradient', 'eigen', 'sdp'
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.method = method
        self.device = device
        
        # Инициализация матрицы A
        if method == 'gradient' and device == 'cuda':
            self.A = torch.randn(output_dim, input_dim, device=device) * 0.01
            self.A.requires_grad_(True)
        else:
            self.A = np.random.randn(output_dim, input_dim) * 0.01
            
        self.M = None  # Матрица M = A^T A
        self.history = {'loss': [], 'rmse': [], 'correlation': []}
        
    def prepare_data(self, X, distances_gt):
        """
        Подготовка данных: создание пар и вычисление целевых расстояний
        
        Parameters:
        -----------
        X : np.ndarray (n_samples, input_dim)
            Входные эмбеддинги
        distances_gt : np.ndarray (n_samples, n_samples) или (n_pairs,)
            Матрица целевых расстояний (например, ΔE в CAM16)
            
        Returns:
        --------
        pairs : list of tuples
            Список пар индексов (i, j)
        diffs : np.ndarray (n_pairs, input_dim)
            Разности векторов для каждой пары
        targets : np.ndarray (n_pairs,)
            Целевые квадраты расстояний
        """
        n = X.shape[0]
        
        # Если distances_gt - матрица, преобразуем в вектор верхнего треугольника
        if distances_gt.ndim == 2 and distances_gt.shape[0] == distances_gt.shape[1]:
            # Берем верхний треугольник без диагонали
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            targets = distances_gt[np.triu_indices(n, k=1)]
        else:
            # Предполагаем, что distances_gt уже вектор
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            targets = distances_gt
            
        targets_squared = targets ** 2  # Работаем с квадратами расстояний
        
        # Вычисляем разности для каждой пары
        diffs = np.array([X[i] - X[j] for i, j in pairs])
        
        return pairs, diffs, targets_squared
    
    def solve_via_eigen(self, X, distances_gt, reg_lambda=1e-6):
        """
        Решение через линеаризацию и собственное разложение (метод 3B)
        
        Parameters:
        -----------
        X : np.ndarray (n_samples, input_dim)
            Входные эмбеддинги
        distances_gt : np.ndarray (n_samples, n_samples)
            Матрица целевых расстояний
        reg_lambda : float
            Коэффициент регуляризации
            
        Returns:
        --------
        A : np.ndarray (output_dim, input_dim)
            Найденная матрица преобразования
        """
        n = X.shape[0]
        k = self.input_dim
        
        # 1. Создаем систему линейных уравнений для vec(M)
        print("Строим систему уравнений для M...")
        B_list = []
        d_list = []
        
        # Берем подвыборку пар для больших наборов данных
        max_pairs = min(10000, n*(n-1)//2)  # Ограничиваем число пар
        indices = np.random.choice(n*(n-1)//2, max_pairs, replace=False)
        
        # Преобразуем линейные индексы в пары (i, j)
        triu_indices = np.triu_indices(n, k=1)
        selected_pairs_i = triu_indices[0][indices]
        selected_pairs_j = triu_indices[1][indices]
        
        for idx in range(max_pairs):
            i, j = selected_pairs_i[idx], selected_pairs_j[idx]
            diff = X[i] - X[j]
            # Векторизация внешнего произведения
            B_vec = np.outer(diff, diff).flatten()
            B_list.append(B_vec)
            d_list.append(distances_gt[i, j] ** 2)
        
        B = np.vstack(B_list)  # (n_pairs, k^2)
        d = np.array(d_list)   # (n_pairs,)
        
        # 2. Решаем МНК: min ||B * vec(M) - d||^2
        print("Решаем систему МНК...")
        # Добавляем регуляризацию
        B_reg = np.vstack([B, np.sqrt(reg_lambda) * np.eye(k*k)])
        d_reg = np.concatenate([d, np.zeros(k*k)])
        
        # Решаем нормальные уравнения
        vec_M = np.linalg.lstsq(B_reg, d_reg, rcond=None)[0]
        
        # 3. Преобразуем вектор обратно в матрицу
        M_hat = vec_M.reshape((k, k))
        
        # 4. Симметризуем
        M_hat = (M_hat + M_hat.T) / 2
        
        # 5. Проекция на PSD конус через собственное разложение
        print("Проецируем на PSD конус...")
        eigvals, eigvecs = np.linalg.eigh(M_hat)
        
        # Обнуляем отрицательные собственные значения
        eigvals_pos = np.maximum(eigvals, 0)
        
        # Добавляем небольшой шум для численной устойчивости
        eigvals_pos += 1e-8
        
        # 6. Факторизация M = A^T A
        print("Факторизуем M...")
        # Выбираем output_dim наибольших собственных значений
        idx_sorted = np.argsort(eigvals_pos)[::-1]
        eigvals_sorted = eigvals_pos[idx_sorted]
        eigvecs_sorted = eigvecs[:, idx_sorted]
        
        # Берем только output_dim компонент
        eigvals_top = eigvals_sorted[:self.output_dim]
        eigvecs_top = eigvecs_sorted[:, :self.output_dim]
        
        # A = sqrt(Λ) * V^T
        sqrt_eigvals = np.sqrt(eigvals_top)
        A = np.diag(sqrt_eigvals) @ eigvecs_top.T
        
        self.A = A
        self.M = M_hat
        
        return A
    
    def solve_via_gradient(self, X, distances_gt, n_epochs=1000, lr=0.01, 
                          batch_size=1000, reg_lambda=1e-4):
        """
        Решение через градиентный спуск (метод 3C)
        
        Parameters:
        -----------
        X : np.ndarray (n_samples, input_dim)
            Входные эмбеддинги
        distances_gt : np.ndarray (n_samples, n_samples)
            Матрица целевых расстояний
        n_epochs : int
            Число эпох обучения
        lr : float
            Скорость обучения
        batch_size : int
            Размер батча
        reg_lambda : float
            Коэффициент регуляризации
            
        Returns:
        --------
        A : np.ndarray (output_dim, input_dim)
            Найденная матрица преобразования
        """
        n = X.shape[0]
        
        # Подготовка данных
        pairs, diffs, targets_squared = self.prepare_data(X, distances_gt)
        n_pairs = len(pairs)
        
        # Преобразуем в тензоры PyTorch если используем GPU
        if self.device == 'cuda':
            X_tensor = torch.FloatTensor(X).to(self.device)
            diffs_tensor = torch.FloatTensor(diffs).to(self.device)
            targets_tensor = torch.FloatTensor(np.sqrt(targets_squared)).to(self.device)  # Без квадрата!
        else:
            X_tensor = torch.FloatTensor(X)
            diffs_tensor = torch.FloatTensor(diffs)
            targets_tensor = torch.FloatTensor(np.sqrt(targets_squared))
        
        # Оптимизатор
        optimizer = optim.Adam([self.A], lr=lr)
        
        print(f"Начинаем градиентный спуск на {n_epochs} эпох...")
        
        for epoch in range(n_epochs):
            total_loss = 0
            
            # Мини-батчи
            for batch_start in range(0, n_pairs, batch_size):
                batch_end = min(batch_start + batch_size, n_pairs)
                
                # Берем батч
                batch_diffs = diffs_tensor[batch_start:batch_end]
                batch_targets = targets_tensor[batch_start:batch_end]
                
                # Прямой проход
                transformed_diffs = self.A @ batch_diffs.T  # (m, batch_size)
                pred_distances = torch.norm(transformed_diffs, dim=0)  # (batch_size,)
                
                # Loss: MSE между предсказанными и истинными расстояниями
                loss = torch.mean((pred_distances - batch_targets) ** 2)
                
                # Регуляризация
                reg_loss = reg_lambda * torch.norm(self.A) ** 2
                total_loss_batch = loss + reg_loss
                
                # Обратный проход
                optimizer.zero_grad()
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item() * (batch_end - batch_start)
            
            # Вычисляем метрики каждые 100 эпох
            if epoch % 100 == 0 or epoch == n_epochs - 1:
                with torch.no_grad():
                    # Преобразуем все данные
                    X_transformed = (self.A @ X_tensor.T).T
                    
                    # Вычисляем все попарные расстояния
                    all_distances_pred = torch.cdist(X_transformed, X_transformed, p=2)
                    all_distances_pred = all_distances_pred.cpu().numpy()
                    
                    # Извлекаем верхний треугольник
                    pred_flat = all_distances_pred[np.triu_indices(n, k=1)]
                    gt_flat = np.sqrt(targets_squared)  # targets_squared это квадраты!
                    
                    # Метрики
                    rmse = np.sqrt(np.mean((pred_flat - gt_flat) ** 2))
                    correlation = np.corrcoef(pred_flat, gt_flat)[0, 1]
                    
                    self.history['loss'].append(total_loss / n_pairs)
                    self.history['rmse'].append(rmse)
                    self.history['correlation'].append(correlation)
                    
                    print(f"Epoch {epoch:4d} | Loss: {total_loss/n_pairs:.6f} | "
                          f"RMSE: {rmse:.6f} | Corr: {correlation:.4f}")
        
        # Преобразуем обратно в numpy
        if self.device == 'cuda':
            self.A = self.A.cpu().detach().numpy()
        else:
            self.A = self.A.detach().numpy()
            
        return self.A
    
    def solve_via_sdp(self, X, distances_gt):
        """
        Решение через Semidefinite Programming (точное, но медленное)
        Требует установки CVXPY
        """
        n = X.shape[0]
        k = self.input_dim
        
        # Создаем переменную M (симметричную PSD)
        M = cp.Variable((k, k), symmetric=True)
        
        # Создаем список ограничений
        constraints = [M >> 0]  # PSD ограничение
        
        # Создаем целевую функцию
        objective = 0
        pair_count = 0
        
        # Ограничиваем число пар для производительности
        max_pairs = min(1000, n*(n-1)//2)
        indices = np.random.choice(n*(n-1)//2, max_pairs, replace=False)
        triu_indices = np.triu_indices(n, k=1)
        
        for idx in indices:
            i, j = triu_indices[0][idx], triu_indices[1][idx]
            diff = X[i] - X[j]
            # quad_form = diff^T M diff
            quad_form = cp.quad_form(diff, M)
            target = distances_gt[i, j] ** 2
            objective += cp.square(quad_form - target)
            pair_count += 1
        
        # Добавляем регуляризацию
        objective += 1e-6 * cp.trace(M)
        
        # Решаем задачу
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.MOSEK, verbose=True)
        
        # Получаем M
        M_opt = M.value
        
        # Факторизуем
        eigvals, eigvecs = np.linalg.eigh(M_opt)
        eigvals_pos = np.maximum(eigvals, 0)
        idx_sorted = np.argsort(eigvals_pos)[::-1]
        eigvals_top = eigvals_pos[idx_sorted][:self.output_dim]
        eigvecs_top = eigvecs[:, idx_sorted][:, :self.output_dim]
        
        A = np.diag(np.sqrt(eigvals_top)) @ eigvecs_top.T
        self.A = A
        self.M = M_opt
        
        return A
    
    def fit(self, X, distances_gt, method=None, **kwargs):
        """
        Основной метод обучения
        
        Parameters:
        -----------
        X : np.ndarray (n_samples, input_dim)
            Входные эмбеддинги
        distances_gt : np.ndarray (n_samples, n_samples)
            Матрица целевых расстояний
        method : str или None
            Метод оптимизации (переопределяет self.method)
        **kwargs : dict
            Параметры для конкретного метода
            
        Returns:
        --------
        self
        """
        if method is None:
            method = self.method
        
        print(f"Обучение метрики методом: {method}")
        print(f"Размерность: {self.input_dim} -> {self.output_dim}")
        print(f"Число образцов: {X.shape[0]}")
        
        if method == 'eigen':
            self.A = self.solve_via_eigen(X, distances_gt, **kwargs)
        elif method == 'gradient':
            self.A = self.solve_via_gradient(X, distances_gt, **kwargs)
        elif method == 'sdp':
            self.A = self.solve_via_sdp(X, distances_gt, **kwargs)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
        
        return self
    
    def transform(self, X):
        """
        Преобразование эмбеддингов
        
        Parameters:
        -----------
        X : np.ndarray (n_samples, input_dim)
            Входные эмбеддинги
            
        Returns:
        --------
        X_transformed : np.ndarray (n_samples, output_dim)
            Преобразованные эмбеддинги
        """
        if isinstance(self.A, torch.Tensor):
            X_tensor = torch.FloatTensor(X).to(self.device)
            X_transformed = (self.A @ X_tensor.T).T
            if self.device == 'cuda':
                return X_transformed.cpu().detach().numpy()
            else:
                return X_transformed.detach().numpy()
        else:
            return (self.A @ X.T).T
    
    def evaluate(self, X, distances_gt, verbose=True):
        """
        Оценка качества обученной метрики
        
        Parameters:
        -----------
        X : np.ndarray (n_samples, input_dim)
            Входные эмбеддинги
        distances_gt : np.ndarray (n_samples, n_samples)
            Матрица целевых расстояний
        verbose : bool
            Выводить ли подробную информацию
            
        Returns:
        --------
        metrics : dict
            Словарь с метриками качества
        """
        # Преобразуем данные
        X_transformed = self.transform(X)
        n = X.shape[0]
        
        # Вычисляем предсказанные расстояния
        distances_pred = squareform(pdist(X_transformed, 'euclidean'))
        
        # Извлекаем верхние треугольники
        gt_flat = distances_gt[np.triu_indices(n, k=1)]
        pred_flat = distances_pred[np.triu_indices(n, k=1)]
        
        # Вычисляем метрики
        mse = np.mean((pred_flat - gt_flat) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_flat - gt_flat))
        correlation = np.corrcoef(pred_flat, gt_flat)[0, 1]
        
        # Процент ошибок менее 1 JND (Just Noticeable Difference)
        # Для цветов обычно JND ≈ 2.3 ΔE
        jnd_threshold = 2.3
        within_jnd = np.mean(np.abs(pred_flat - gt_flat) < jnd_threshold) * 100
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation,
            'within_jnd_percent': within_jnd,
            'mse': mse
        }
        
        if verbose:
            print("\n" + "="*60)
            print("ОЦЕНКА КАЧЕСТВА МЕТРИКИ")
            print("="*60)
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"Корреляция Пирсона: {correlation:.4f}")
            print(f"В пределах JND ({jnd_threshold} ΔE): {within_jnd:.1f}%")
            print(f"MSE: {mse:.4f}")
            
            # Анализ ошибок по диапазонам расстояний
            print("\nАнализ ошибок по диапазонам расстояний:")
            bins = [0, 5, 10, 20, 50, 100, np.inf]
            bin_labels = ['0-5', '5-10', '10-20', '20-50', '50-100', '100+']
            
            for i in range(len(bins)-1):
                mask = (gt_flat >= bins[i]) & (gt_flat < bins[i+1])
                if np.sum(mask) > 0:
                    rmse_bin = np.sqrt(np.mean((pred_flat[mask] - gt_flat[mask]) ** 2))
                    print(f"  ΔE ∈ [{bins[i]:3d}, {bins[i+1]:3d}]: "
                          f"RMSE = {rmse_bin:.3f}, n = {np.sum(mask)}")
        
        return metrics
    
    def plot_results(self, X, distances_gt):
        """
        Визуализация результатов
        
        Parameters:
        -----------
        X : np.ndarray
            Входные эмбеддинги
        distances_gt : np.ndarray
            Матрица целевых расстояний
        """
        X_transformed = self.transform(X)
        n = X.shape[0]
        
        # Извлекаем расстояния
        gt_flat = distances_gt[np.triu_indices(n, k=1)]
        pred_flat = squareform(pdist(X_transformed, 'euclidean'))[np.triu_indices(n, k=1)]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Сравнение расстояний
        ax1 = axes[0, 0]
        scatter = ax1.scatter(gt_flat, pred_flat, s=1, alpha=0.3)
        max_val = max(gt_flat.max(), pred_flat.max())
        ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Идеально')
        ax1.set_xlabel('Истинные расстояния (ΔE)')
        ax1.set_ylabel('Предсказанные расстояния')
        ax1.set_title(f'Корреляция: {np.corrcoef(gt_flat, pred_flat)[0,1]:.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Ошибки
        ax2 = axes[0, 1]
        errors = pred_flat - gt_flat
        ax2.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--')
        ax2.set_xlabel('Ошибка (предсказано - истинное)')
        ax2.set_ylabel('Частота')
        ax2.set_title(f'Распределение ошибок (MAE={np.mean(np.abs(errors)):.3f})')
        
        # 3. Относительные ошибки
        ax3 = axes[0, 2]
        rel_errors = np.abs(errors) / (gt_flat + 1e-6)
        ax3.hist(rel_errors, bins=50, alpha=0.7, edgecolor='black', range=(0, 2))
        ax3.set_xlabel('Относительная ошибка')
        ax3.set_ylabel('Частота')
        ax3.set_title('Распределение относительных ошибок')
        
        # 4. История обучения (если есть)
        if self.history['loss']:
            ax4 = axes[1, 0]
            epochs = range(len(self.history['loss']))
            ax4.plot(epochs, self.history['loss'], 'b-', label='Loss')
            ax4.set_xlabel('Эпоха (x100)')
            ax4.set_ylabel('Loss', color='b')
            ax4.tick_params(axis='y', labelcolor='b')
            ax4.set_title('История обучения')
            
            ax5 = ax4.twinx()
            ax5.plot(epochs, self.history['correlation'], 'r-', label='Корреляция')
            ax5.set_ylabel('Корреляция', color='r')
            ax5.tick_params(axis='y', labelcolor='r')
            ax4.legend(loc='upper left')
            ax5.legend(loc='upper right')
        
        # 5. Собственные значения матрицы M
        if self.M is not None:
            ax6 = axes[1, 1]
            eigvals = np.linalg.eigvalsh(self.M)
            ax6.plot(range(1, len(eigvals)+1), np.sort(eigvals)[::-1], 'o-')
            ax6.set_xlabel('Номер собственного значения')
            ax6.set_ylabel('Значение')
            ax6.set_title('Собственные значения M')
            ax6.grid(True, alpha=0.3)
            
            # Вычисляем объясненную дисперсию
            explained_variance = np.cumsum(np.sort(eigvals)[::-1]) / np.sum(eigvals)
            ax7 = axes[1, 2]
            ax7.plot(range(1, len(explained_variance)+1), explained_variance, 's-')
            ax7.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='95%')
            ax7.axhline(0.99, color='green', linestyle='--', alpha=0.5, label='99%')
            ax7.set_xlabel('Число компонент')
            ax7.set_ylabel('Объясненная дисперсия')
            ax7.set_title('Объясненная дисперсия')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Дополнительная визуализация: тепловая карта ошибок
        if n <= 100:  # Только для небольших наборов
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
            
            distances_pred = squareform(pdist(X_transformed, 'euclidean'))
            error_matrix = np.abs(distances_pred - distances_gt)
            
            im1 = ax1.imshow(distances_gt, cmap='viridis')
            ax1.set_title('Истинные расстояния')
            plt.colorbar(im1, ax=ax1)
            
            im2 = ax2.imshow(distances_pred, cmap='viridis')
            ax2.set_title('Предсказанные расстояния')
            plt.colorbar(im2, ax=ax2)
            
            im3 = ax3.imshow(error_matrix, cmap='hot')
            ax3.set_title('Абсолютные ошибки')
            plt.colorbar(im3, ax=ax3)
            
            plt.tight_layout()
            plt.show()


# Пример использования с вашими данными
def run_metric_learning_example(embeddings, cam_distances, layer_name="LM"):
    """
    Пример использования метрического обучения для ваших данных
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Эмбеддинги из вашей модели
    cam_distances : np.ndarray
        Матрица расстояний в CAM16-LCD (ΔE)
    layer_name : str
        Имя слоя для вывода
    """
    print(f"\n{'='*60}")
    print(f"METRIC LEARNING: {layer_name} LAYER")
    print(f"{'='*60}")
    
    # Разделение на train/test
    n_samples = embeddings.shape[0]
    train_idx, test_idx = train_test_split(
        np.arange(n_samples), test_size=0.3, random_state=42
    )
    
    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]
    
    # Соответствующие подматрицы расстояний
    distances_train = cam_distances[np.ix_(train_idx, train_idx)]
    distances_test = cam_distances[np.ix_(test_idx, test_idx)]
    
    # 1. Метод 1: Быстрый (Eigen)
    print("\n1. Метод Eigen (быстрый):")
    learner1 = ColorMetricLearner(
        input_dim=embeddings.shape[1], 
        output_dim=3,
        method='eigen'
    )
    learner1.fit(X_train, distances_train, reg_lambda=1e-4)
    metrics1 = learner1.evaluate(X_test, distances_test)
    
    # 2. Метод 2: Градиентный (более точный)
    print("\n2. Метод Gradient (точный):")
    learner2 = ColorMetricLearner(
        input_dim=embeddings.shape[1],
        output_dim=3,
        method='gradient'
    )
    learner2.fit(X_train, distances_train, n_epochs=500, lr=0.01, batch_size=1000)
    metrics2 = learner2.evaluate(X_test, distances_test)
    
    # 3. Сравнение методов
    print("\n" + "="*60)
    print("СРАВНЕНИЕ МЕТОДОВ:")
    print("="*60)
    print(f"{'Метод':<15} {'RMSE':<10} {'MAE':<10} {'Corr':<10} {'<JND%':<10}")
    print("-"*60)
    print(f"{'Eigen':<15} {metrics1['rmse']:<10.4f} {metrics1['mae']:<10.4f} "
          f"{metrics1['correlation']:<10.4f} {metrics1['within_jnd_percent']:<10.1f}")
    print(f"{'Gradient':<15} {metrics2['rmse']:<10.4f} {metrics2['mae']:<10.4f} "
          f"{metrics2['correlation']:<10.4f} {metrics2['within_jnd_percent']:<10.1f}")
    
    # Визуализация лучшего метода
    print(f"\nВизуализация лучшего метода...")
    best_learner = learner2 if metrics2['rmse'] < metrics1['rmse'] else learner1
    best_learner.plot_results(X_test, distances_test)
    
    # Анализ матрицы преобразования
    A = best_learner.A
    print(f"\nАнализ матрицы A ({layer_name}):")
    print(f"Размер: {A.shape}")
    print(f"Норма Фробениуса: {np.linalg.norm(A):.4f}")
    
    # Сингулярные значения
    U, s, Vt = np.linalg.svd(A)
    print(f"Сингулярные значения: {s}")
    print(f"Обусловленность: {s[0]/s[-1] if s[-1] > 0 else 'inf':.2f}")
    
    # Объясненная дисперсия
    explained_var = np.cumsum(s**2) / np.sum(s**2)
    print(f"Объясненная дисперсия по компонентам:")
    for i, var in enumerate(explained_var):
        print(f"  Компонента {i+1}: {var:.3f}")
    
    return best_learner, metrics1, metrics2


# Интеграция с вашим существующим кодом
def compare_linear_and_metric_learning(embeddings, colors_cam, layer_name="LM"):
    """
    Сравнение линейной регрессии и метрического обучения
    """
    # 1. Линейная регрессия (ваш существующий подход)
    print(f"\n{'='*60}")
    print(f"LINEAR REGRESSION: {layer_name}")
    print(f"{'='*60}")
    
    # Ваш код линейной регрессии здесь...
    # (используйте вашу функцию linear_color_mapping)
    
    # 2. Метрическое обучение
    # Вычисляем матрицу расстояний в CAM16
    cam_distances = squareform(pdist(colors_cam, 'euclidean'))
    
    # Запускаем метрическое обучение
    best_learner, metrics_eigen, metrics_gradient = run_metric_learning_example(
        embeddings, cam_distances, layer_name
    )
    
    # 3. Сравнение предсказанных цветов
    # Преобразуем через метрическое обучение
    colors_metric = best_learner.transform(embeddings)
    
    # Вычисляем расстояния для сравнения
    distances_linear = pdist(colors_pred_linear, 'euclidean')  # от линейной регрессии
    distances_metric = pdist(colors_metric, 'euclidean')
    distances_true = pdist(colors_cam, 'euclidean')
    
    print(f"\n{'='*60}")
    print(f"FINAL COMPARISON: {layer_name}")
    print(f"{'='*60}")
    
    # Корреляции
    corr_linear = np.corrcoef(distances_linear, distances_true)[0, 1]
    corr_metric = np.corrcoef(distances_metric, distances_true)[0, 1]
    
    # RMSE
    rmse_linear = np.sqrt(np.mean((distances_linear - distances_true) ** 2))
    rmse_metric = np.sqrt(np.mean((distances_metric - distances_true) ** 2))
    
    print(f"Linear Regression: Corr = {corr_linear:.4f}, RMSE = {rmse_linear:.4f}")
    print(f"Metric Learning:   Corr = {corr_metric:.4f}, RMSE = {rmse_metric:.4f}")
    
    improvement = (rmse_linear - rmse_metric) / rmse_linear * 100
    print(f"\nУлучшение RMSE: {improvement:.1f}%")
    
    return {
        'linear': {'correlation': corr_linear, 'rmse': rmse_linear},
        'metric': {'correlation': corr_metric, 'rmse': rmse_metric},
        'learner': best_learner,
        'colors_metric': colors_metric
    }


# Основной скрипт
if __name__ == "__main__":
    from utils.analyze.munsell_analyze import MunsellEmbeddingsAnalyzer
    from colour.models import XYZ_to_CAM16LCD
    from colour import xyY_to_XYZ
    
    analyzer = MunsellEmbeddingsAnalyzer('data/embeddings/qwen2.5_7B/munsell_colors_describe', 'data/colors/munsell_colors/munsell_manifest.csv')
    data = analyzer.chain_loader.get_all_available_embeddings()
    lm_matrix_all = data['lm_pooled']
    vision_matrix_all = data['vl_pooled']
    color_meta_all = data['metadata']

    keys = sorted(color_meta_all.keys(), key=lambda k: int(k))
    colors_xyY_all = np.array([
        [color_meta_all[k]['xyY']['x'],
        color_meta_all[k]['xyY']['y'],
        color_meta_all[k]['xyY']['Y']]
        for k in keys
        ], dtype=float)
    
    colors_cam = np.array([XYZ_to_CAM16LCD(xyY_to_XYZ(c)) for c in colors_xyY_all])
    print(colors_xyY_all.shape) 
    # colors_cam = ... # ваши CAM16-LCD цвета
    
    # Для LM слоя
    results_lm = compare_linear_and_metric_learning(
        lm_matrix_all, colors_cam, "LM"
    )
    
    # Для Vision слоя
    # results_vision = compare_linear_and_metric_learning(
    #     vision_matrix_all, colors_cam, "Vision"
    # )
    
    print("Готово! Класс ColorMetricLearner реализует все методы из вашего плана.")
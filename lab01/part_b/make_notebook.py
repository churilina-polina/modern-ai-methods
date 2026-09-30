import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Лабораторная работа 1, часть 2. Transfer Learning
**Задача:** классификация пород кошек и собак (Oxford-IIIT Pet, 37 классов) с помощью предобученной сети **MobileNetV2**.

Две стратегии:
1. **Feature Extraction:** веса базовой сети заморожены, обучается только новый выходной слой.
2. **Fine-tuning:** размораживаются верхние блоки базовой сети и дообучаются с маленьким learning rate.""")

code("""import os, time, json, copy, random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

try:  # на macOS с Python с python.org не всегда подхватываются SSL-сертификаты
    import certifi; os.environ.setdefault('SSL_CERT_FILE', certifi.where())
except ImportError:
    pass

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Устройство: GPU (CUDA в Colab), Apple Silicon (MPS) или CPU
device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available() else 'cpu')
print('Device:', device, '| torch', torch.__version__)""")

md("""## 1. Датасет Oxford-IIIT Pet
Около 7 400 фотографий 37 пород (12 пород кошек, 25 пород собак), примерно по 200 фото на класс.
Используем официальное разбиение: `trainval` (3680) делим на train/val в пропорции 90/10, `test` (3669) берём для итоговой оценки.""")

code("""DATA_DIR = './data'
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # нормализация ImageNet

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

full_train = datasets.OxfordIIITPet(DATA_DIR, split='trainval', transform=train_tf, download=True)
full_val   = datasets.OxfordIIITPet(DATA_DIR, split='trainval', transform=eval_tf,  download=True)
test_set   = datasets.OxfordIIITPet(DATA_DIR, split='test',     transform=eval_tf,  download=True)
class_names = full_train.classes
num_classes = len(class_names)

idx = np.random.RandomState(SEED).permutation(len(full_train))
n_val = int(0.1 * len(idx))
train_set = Subset(full_train, idx[n_val:])
val_set   = Subset(full_val,   idx[:n_val])

BATCH = 32
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=64,    shuffle=False, num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=64,    shuffle=False, num_workers=0)

print(f'Классов: {num_classes}')
print(f'Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}')
print(class_names)""")

code("""# Примеры изображений из датасета
def denorm(t):
    return (t.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)

fig, axes = plt.subplots(2, 6, figsize=(15, 5.5))
for ax, i in zip(axes.flat, np.random.RandomState(1).choice(len(test_set), 12, replace=False)):
    img, label = test_set[i]
    ax.imshow(denorm(img)); ax.set_title(class_names[label], fontsize=9); ax.axis('off')
plt.suptitle('Примеры изображений Oxford-IIIT Pet'); plt.tight_layout(); plt.show()""")

md("""## 2. Выбор архитектуры: MobileNetV2
* **Лёгкая:** 3.5 млн параметров (у ResNet-50 около 25.6 млн, у EfficientNet-B0 около 5.3 млн), поэтому быстро обучается даже без мощной видеокарты (на ноутбуке и в бесплатном Colab).
* **Точная:** на ImageNet top-1 около 72% (веса IMAGENET1K_V2), этого достаточно, чтобы признаки хорошо переносились на новые задачи.
* **Архитектура:** *inverted residual* блоки с *depthwise separable* свёртками и линейным bottleneck. Depthwise-свёртка обрабатывает каждый канал отдельно, pointwise-свёртка 1×1 смешивает каналы. Вычислений примерно в 8–9 раз меньше, чем у обычной свёртки.
* ImageNet содержит много классов кошек и собак, поэтому предобученные признаки хорошо подходят именно под нашу задачу.""")

code("""def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    # Заменяем выходной слой: 1000 классов ImageNet -> 37 пород
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model.to(device)

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

m = build_model()
print(m.classifier)
print('Блоков в model.features:', len(m.features))
print('Всего параметров: %d' % count_params(m)[0])
del m""")

code("""def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n

def fit(model, optimizer, epochs, scheduler=None, tag=''):
    criterion = nn.CrossEntropyLoss()
    hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc, best_state = 0, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, criterion, optimizer)
        vl, va = run_epoch(model, val_loader, criterion)
        if scheduler: scheduler.step()
        for k, v in zip(hist, (tl, ta, vl, va)): hist[k].append(v)
        if va > best_acc:
            best_acc, best_state = va, copy.deepcopy(model.state_dict())
        print(f'[{tag}] epoch {ep}/{epochs}  train loss {tl:.3f} acc {ta:.3f} | '
              f'val loss {vl:.3f} acc {va:.3f} | {time.time()-t0:.0f}s')
    model.load_state_dict(best_state)
    return hist

@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()

def metrics(y_true, y_pred):
    return {
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall':    recall_score(y_true, y_pred, average='macro', zero_division=0),
        'F1':        f1_score(y_true, y_pred, average='macro', zero_division=0),
    }

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Predicted label'); ax.set_ylabel('True label'); ax.set_title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04); plt.tight_layout(); plt.show()

def top_confusions(y_true, y_pred, k=5):
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm.ravel())[::-1][:k]
    for f in flat:
        i, j = divmod(f, num_classes)
        print(f'  {class_names[i]:28s} -> {class_names[j]:28s}: {cm[i, j]} ошибок')""")

md("""## 3. Стратегия 1: Feature Extraction
Замораживаем **все** веса сверточной части (`requires_grad = False`), так что она работает как фиксированный извлекатель признаков (вектор 1280 чисел). Обучаем только новый слой `Linear(1280 → 37)`.""")

code("""fe_model = build_model()
for p in fe_model.features.parameters():
    p.requires_grad = False

total, trainable = count_params(fe_model)
print(f'Обучаемых параметров: {trainable:,} из {total:,} ({100*trainable/total:.2f}%)')

EPOCHS_FE = 6
opt_fe = torch.optim.Adam(fe_model.classifier.parameters(), lr=1e-3)
t0 = time.time()
hist_fe = fit(fe_model, opt_fe, EPOCHS_FE, tag='FE')
time_fe = time.time() - t0
print(f'Время обучения: {time_fe/60:.1f} мин')""")

code("""y_true, y_pred_fe = predict(fe_model, test_loader)
m_fe = metrics(y_true, y_pred_fe)
print('Feature Extraction, метрики на тестовой выборке:')
for k, v in m_fe.items(): print(f'  {k:10s} = {100*v:.2f}%')
print('\\nСамые частые ошибки:'); top_confusions(y_true, y_pred_fe)""")

code("""plot_confusion(y_true, y_pred_fe, 'Confusion matrix: Feature Extraction (MobileNetV2)')""")

md("""## 4. Стратегия 2: Fine-tuning
Берём модель после Feature Extraction (новая «голова» уже обучена, поэтому большие случайные градиенты не испортят предобученные веса). **Размораживаем верхние блоки** `features[14:]` (последние 5 блоков из 19, самые «специфичные» признаки) и дообучаем с **маленьким learning rate**: `1e-4` для базовой сети и `1e-3` для головы, с косинусным затуханием.
Нижние блоки (края, текстуры) остаются замороженными: эти признаки универсальны.""")

code("""ft_model = copy.deepcopy(fe_model)
UNFREEZE_FROM = 14
for i, block in enumerate(ft_model.features):
    for p in block.parameters():
        p.requires_grad = i >= UNFREEZE_FROM

# BatchNorm в замороженных блоках оставляем в режиме eval, чтобы не сбивать их статистику
def freeze_bn(model):
    for i, block in enumerate(model.features):
        if i < UNFREEZE_FROM:
            for mod in block.modules():
                if isinstance(mod, nn.BatchNorm2d): mod.eval()

total, trainable = count_params(ft_model)
print(f'Обучаемых параметров: {trainable:,} из {total:,} ({100*trainable/total:.2f}%)')

EPOCHS_FT = 6
opt_ft = torch.optim.Adam([
    {'params': [p for p in ft_model.features.parameters() if p.requires_grad], 'lr': 1e-4},
    {'params': ft_model.classifier.parameters(), 'lr': 1e-3},
])
sched_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=EPOCHS_FT)

_orig_train = ft_model.train
def _train(mode=True):
    _orig_train(mode)
    if mode: freeze_bn(ft_model)
    return ft_model
ft_model.train = _train

t0 = time.time()
hist_ft = fit(ft_model, opt_ft, EPOCHS_FT, scheduler=sched_ft, tag='FT')
time_ft = time.time() - t0
print(f'Время обучения: {time_ft/60:.1f} мин')""")

code("""_, y_pred_ft = predict(ft_model, test_loader)
m_ft = metrics(y_true, y_pred_ft)
print('Fine-tuning, метрики на тестовой выборке:')
for k, v in m_ft.items(): print(f'  {k:10s} = {100*v:.2f}%')
print('\\nСамые частые ошибки:'); top_confusions(y_true, y_pred_ft)""")

code("""plot_confusion(y_true, y_pred_ft, 'Confusion matrix: Fine-tuning (MobileNetV2)')""")

md("""## 5. Сравнение стратегий""")

code("""print(f"{'Метрика':12s} {'Feature Extraction':>20s} {'Fine-tuning':>14s} {'Разница':>10s}")
for k in m_fe:
    print(f"{k:12s} {100*m_fe[k]:19.2f}% {100*m_ft[k]:13.2f}% {100*(m_ft[k]-m_fe[k]):+9.2f}%")
print(f"{'Время, мин':12s} {time_fe/60:20.1f} {time_ft/60:14.1f}")
print(f"{'Эпох':12s} {EPOCHS_FE:20d} {str(EPOCHS_FE) + '+' + str(EPOCHS_FT):>14s}")

# Сохраняем результаты (из них собирается текст отчета)
def top_pairs(y_pred, k=5):
    cm = confusion_matrix(y_true, y_pred); np.fill_diagonal(cm, 0)
    return [[class_names[i], class_names[j], int(cm[i, j])]
            for i, j in (divmod(f, num_classes) for f in np.argsort(cm.ravel())[::-1][:k])]
# Порода кошки или собаки: в Oxford-IIIT Pet породы кошек пишутся с заглавной буквы в именах файлов
is_cat = np.zeros(num_classes, bool)
for path, lab in zip(test_set._images, test_set._labels):
    is_cat[lab] = os.path.basename(path)[0].isupper()
cat_dog_err = lambda yp: int((is_cat[y_true] != is_cat[yp]).sum())
f1_cls = f1_score(y_true, y_pred_ft, average=None)
json.dump({'feature_extraction': m_fe, 'fine_tuning': m_ft,
           'time_fe_min': time_fe/60, 'time_ft_min': time_ft/60,
           'epochs_fe': EPOCHS_FE, 'epochs_ft': EPOCHS_FT,
           'params_total': count_params(fe_model)[0],
           'params_fe': sum(p.numel() for p in fe_model.parameters() if p.requires_grad),
           'params_ft': sum(p.numel() for p in ft_model.parameters() if p.requires_grad),
           'errors_fe': int((y_pred_fe != y_true).sum()), 'errors_ft': int((y_pred_ft != y_true).sum()),
           'n_test': int(len(y_true)),
           'cat_dog_errors_fe': cat_dog_err(y_pred_fe), 'cat_dog_errors_ft': cat_dog_err(y_pred_ft),
           'top_fe': top_pairs(y_pred_fe), 'top_ft': top_pairs(y_pred_ft),
           'worst_ft': [[class_names[i], float(f1_cls[i])] for i in np.argsort(f1_cls)[:3]],
           'best_ft': [[class_names[i], float(f1_cls[i])] for i in np.argsort(f1_cls)[::-1][:3]],
           'hist_fe': hist_fe, 'hist_ft': hist_ft},
          open('results.json', 'w'), indent=2, ensure_ascii=False)
print('Ошибок кошка<->собака: FE', cat_dog_err(y_pred_fe), '| FT', cat_dog_err(y_pred_ft))""")

code("""# Метрики обеих стратегий на одном графике
names = list(m_fe)
x = np.arange(len(names)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x - w/2, [100*m_fe[k] for k in names], w, label='Feature Extraction')
b2 = ax.bar(x + w/2, [100*m_ft[k] for k in names], w, label='Fine-tuning')
ax.bar_label(b1, fmt='%.1f', fontsize=9); ax.bar_label(b2, fmt='%.1f', fontsize=9)
ax.set_xticks(x, names); ax.set_ylabel('%')
lo = min(min(m_fe.values()), min(m_ft.values())) * 100
ax.set_ylim(max(0, lo - 10), 100)
ax.set_title('Сравнение стратегий на тестовой выборке'); ax.legend(loc='lower right')
plt.tight_layout(); plt.show()""")

code("""# Кривые обучения: Fine-tuning продолжает Feature Extraction
ep_fe = np.arange(1, EPOCHS_FE + 1)
ep_ft = np.arange(EPOCHS_FE + 1, EPOCHS_FE + EPOCHS_FT + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, key, title in [(axes[0], 'loss', 'Loss'), (axes[1], 'acc', 'Accuracy')]:
    ax.plot(ep_fe, hist_fe['train_' + key], 'o-', c='tab:blue',   label='FE train')
    ax.plot(ep_fe, hist_fe['val_' + key],   's--', c='tab:blue',  label='FE val')
    ax.plot(ep_ft, hist_ft['train_' + key], 'o-', c='tab:orange', label='FT train')
    ax.plot(ep_ft, hist_ft['val_' + key],   's--', c='tab:orange',label='FT val')
    ax.axvline(EPOCHS_FE + 0.5, c='gray', ls=':'); ax.set_xlabel('Эпоха')
    ax.set_title(title); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()""")

code("""# Примеры предсказаний модели после Fine-tuning (зелёный: верно, красный: ошибка)
fig, axes = plt.subplots(3, 6, figsize=(16, 8.5))
for ax, i in zip(axes.flat, np.random.RandomState(7).choice(len(test_set), 18, replace=False)):
    img, label = test_set[i]
    ax.imshow(denorm(img)); ax.axis('off')
    p = y_pred_ft[i]
    ax.set_title(f'{class_names[p]}\\n(true: {class_names[label]})', fontsize=8,
                 color='green' if p == label else 'red')
plt.tight_layout(); plt.show()""")

code("""print(classification_report(y_true, y_pred_ft, target_names=class_names, digits=3))""")

nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, 'transfer_learning.ipynb')
print('ok')

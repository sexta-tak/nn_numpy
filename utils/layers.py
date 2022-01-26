import numpy as np
from collections import OrderedDict
from activation import softmax, relu
from loss import cross_entropy_error
from util import im2col, col2im

class Module:
    def __init__(self, required_grad=True):
        self.params = OrderedDict()
        self.layers = OrderedDict()
        if required_grad:
            self.grad = OrderedDict()
        self.is_params = True
        self.train = True


class Affine(Module):
    def __init__(self, in_features, out_features, weight_init_std=0.01, bias=True):
        super().__init__()

        self.bias = bias
        W = weight_init_std * np.random.randn(in_features, out_features)
        b = np.zeros(out_features) if bias else None
        self.params = {"W":W, "b":b}
        
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.grads = {"W":None, "b":None}

    def forward(self, x):
        # テンソル対応(画像形式のxに対応させる)
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.params["W"]) + self.params["b"]

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.params["W"].T)
        self.grads["W"] = np.dot(self.x.T, dout)
        if self.bias:
            self.grads["b"] = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        
        # 初期値
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ
        self.is_params = False

    def forward(self, x, t):
        """
        順伝播
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        """
        逆伝播
        伝播する値をバッチサイズで割ること
        dout=1は、他のレイヤと同じ使い方ができるように設定しているダミー変数
        """
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

class ReLU:
    def __init__(self):
        self.mask = None
        self.is_params = False

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Convolution(Module):
    def __init__(self, input_ch, output_ch, filter_size, stride, padding, weight_init_std=0.01, bias=True):
        super().__init__()

        self.bias = bias
        W = weight_init_std * np.random.randn(output_ch, input_ch, filter_size, filter_size)
        b = np.zeros(output_ch) if bias else None
        self.params = {"W":W, "b":b}
        self.stride = stride
        self.pad = padding

        # 重み・バイアスパラメータの微分
        self.grads = {"W":None, "b":None}
        self.x = None   
        self.col = None
        self.col_W = None
        self.dcol = None

    def forward(self, x):
        FN, C, FH, FW = self.params["W"].shape
        N, C, H, W = x.shape
        out_h = (H + 2*self.pad - FH) // self.stride + 1 # 出力の高さ(端数は切り捨てる)
        out_w =(W + 2*self.pad - FW) // self.stride + 1# 出力の幅(端数は切り捨てる)

        # 畳み込み演算を効率的に行えるようにするため、入力xを行列colに変形する
        col = im2col(x, FH, FW, self.stride, self.pad)
        
        # 重みフィルターを2次元配列に変形する
        # col_Wの配列形状は、(C*FH*FW, フィルター枚数)
        col_W = self.params["W"].reshape(FN, -1).T

        # 行列の積を計算し、バイアスを足す
        if self.bias:
            out = np.dot(col, col_W) + self.params["b"]
        else:
            out = np.dot(col, col_W)
        
        # 画像形式に戻して、チャンネルの軸を2番目に移動する
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        """
        逆伝播計算
        Affineレイヤと同様の考え方で、逆伝播させる
        dout : 出力層側から伝わってきた勾配(配列形状=(データ数, チャンネル数, 高さ, 幅))
        return : 入力層側へ伝える勾配
        """
        FN, C, FH, FW = self.params["W"].shape
        
        # doutのチャンネル数軸を4番目に移動させ、2次元配列に変形する
        # doutの列数は、チャンネル数(=フィルター数)になる
        # doutの行数は、データ数*doutの高さ*doutの幅になる        
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        # バイアスbは、doutのチャンネル毎に、(データ数*doutの高さ*doutの幅)個の要素を足し合わせる
        if self.bias:
            self.grads["b"] = np.sum(dout, axis=0)
        
        # dWは、入力行列colと行列doutの積になる
        self.grads["W"] = np.dot(self.col.T, dout)
        
        # dWを(フィルター数, チャンネル数, フィルター高さ、フィルター幅)の配列形状に変形する
        self.grads["W"] = self.grads["W"].transpose(1, 0).reshape(FN, C, FH, FW)

        # 入力側の勾配は、doutにフィルターの重みを掛けて求める
        dcol = np.dot(dout, self.col_W.T)
        
        # 勾配を4次元配列(データ数, チャンネル数, 高さ, 幅)に変形する
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad, is_backward=True)

        self.dcol = dcol # 結果を確認するために保持しておく
            
        return dx

class MaxPooling(Module):
    def __init__(self, filter_size=2, stride=2, padding=0):
        super().__init__()

        self.pool_h = filter_size
        self.pool_w = filter_size
        self.stride = stride
        self.pad = padding

        # インスタンス変数の宣言
        self.x = None
        self.arg_max = None
        self.col = None
        self.dcol = None
        
        self.is_params = False

    def forward(self, x):
        """
        順伝播計算
        x : 入力(配列形状=(データ数, チャンネル数, 高さ, 幅))
        """        
        N, C, H, W = x.shape
        
        # 出力サイズ
        out_h = (H  + 2*self.pad - self.pool_h) // self.stride + 1 # 出力の高さ(端数は切り捨てる)
        out_w = (W + 2*self.pad - self.pool_w) // self.stride + 1# 出力の幅(端数は切り捨てる)    
        
        # プーリング演算を効率的に行えるようにするため、2次元配列に変形する
        # パディングする値は、マイナスの無限大にしておく
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad, constant_values=-np.inf)
        
        # チャンネル方向のデータが横に並んでいるので、縦に並べ替える
        # 変形後のcolの配列形状は、(N*out_h*out_w*C, pool_h*pool_w)になる 
        col = col.reshape(-1, self.pool_h*self.pool_w)

        # 最大値のインデックスを求める
        # この結果は、逆伝播計算時に用いる
        arg_max = np.argmax(col, axis=1)
        
        # 最大値を求める
        out = np.max(col, axis=1)
        
        # 画像形式に戻して、チャンネルの軸を2番目に移動する
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        """
        逆伝播計算
        マックスプーリングでは、順伝播計算時に最大値となった場所だけに勾配を伝える
        順伝播計算時に最大値となった場所は、self.arg_maxに保持されている        
        dout : 出力層側から伝わってきた勾配
        return : 入力層側へ伝える勾配
        """        
        
        # doutのチャンネル数軸を4番目に移動させる
        dout = dout.transpose(0, 2, 3, 1)
        
        # プーリング適応領域の要素数(プーリング適応領域の高さ × プーリング適応領域の幅)
        pool_size = self.pool_h * self.pool_w
        
        # 勾配を入れる配列を初期化する
        # dcolの配列形状 : (doutの全要素数, プーリング適応領域の要素数) 
        # doutの全要素数は、dout.size で取得できる
        dcol = np.zeros((dout.size, pool_size))
        
        # 順伝播計算時に最大値となった場所に、doutを配置する
        # dout.flatten()でdoutを1次元配列に変形できる
        dcol[np.arange(dcol.shape[0]), self.arg_max] = dout.flatten()
        
        # 勾配を4次元配列(データ数, チャンネル数, 高さ, 幅)に変形する
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad, is_backward=True)
        
        self.dcol = dcol # 結果を確認するために保持しておく
        
        return dx


class BatchNormalization(Module):
    def __init__(self, input_size, rho=0.9, moving_mean=None, moving_var=None):
        super().__init__()

        gamma = np.ones(input_size) # スケールさせるためのパラメータ, 学習によって更新させる.
        beta = np.zeros(input_size) # シフトさせるためのパラメータ, 学習によって更新させる
        self.params = {"gamma":gamma, "beta":beta}
        self.rho = rho # 移動平均を算出する際に使用する係数

        # 予測時に使用する平均と分散
        self.moving_mean = moving_mean   # muの移動平均
        self.moving_var = moving_var     # varの移動平均
        
        # 計算中に算出される値を保持しておく変数群
        self.batch_size = None
        self.x_mu = None
        self.x_std = None        
        self.std = None
        self.grads = {"gamma":None, "beta":None}

    def forward(self, x):
        """
        順伝播計算
        x :  CNNの場合は4次元、全結合層の場合は2次元  
        """
        if x.ndim == 4:
            """
            画像形式の場合
            """
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1) # NHWCに入れ替え
            x = x.reshape(N*H*W, C) # (N*H*W,C)の2次元配列に変換
            out = self.__forward(x, train_flg)
            out = out.reshape(N, H, W, C)# 4次元配列に変換
            out = out.transpose(0, 3, 1, 2) # 軸をNCHWに入れ替え
        elif x.ndim == 2:
            """
            画像形式以外の場合
            """
            out = self.__forward(x)           
            
        return out
            
    def __forward(self, x, epsilon=1e-8):
        """
        x : 入力. N×Dの行列. Nはバッチサイズ. Dは手前の層のノード数
        """
        if (self.moving_mean is None) or (self.moving_var is None):
            N, D = x.shape
            self.moving_mean = np.zeros(D)
            self.moving_var = np.zeros(D)
                        
        if self.train:
            """
            学習時
            """
            # 入力xについて、nの方向に平均値を算出. 
            mu = x.mean(axis=0) # 要素数d個のベクトル
            
            # 入力xから平均値を引く
            x_mu = x - mu   # n*d行列
            
            # 入力xの分散を求める
            var = np.mean(x_mu**2, axis=0)  # 要素数d個のベクトル
            
            # 入力xの標準偏差を求める(epsilonを足してから標準偏差を求める)
            std = np.sqrt(var + epsilon)  # 要素数d個のベクトル
            
            # 標準化
            x_std = x_mu / std  # n*d行列
            
            # 値を保持しておく
            self.batch_size = x.shape[0]
            self.x_mu = x_mu
            self.x_std = x_std
            self.std = std
            self.moving_mean = self.rho * self.moving_mean + (1-self.rho) * mu
            self.moving_var = self.rho * self.moving_var + (1-self.rho) * var            
        else:
            """
            予測時
            """
            x_mu = x - self.moving_mean # n*d行列
            x_std = x_mu / np.sqrt(self.moving_var + epsilon) # n*d行列
            
        # gammaでスケールし、betaでシフトさせる
        out = self.params["gamma"] * x_std + self.params["beta"] # N*D行列
        return out

    def backward(self, dout):
        """
        逆伝播計算
        dout : CNNの場合は4次元、全結合層の場合は2次元  
        """
        if dout.ndim == 4:
            """
            画像形式の場合
            """            
            N, C, H, W = dout.shape
            dout = dout.transpose(0, 2, 3, 1) # NHWCに入れ替え
            dout = dout.reshape(N*H*W, C) # (N*H*W,C)の2次元配列に変換
            dx = self.__backward(dout)
            dx = dx.reshape(N, H, W, C)# 4次元配列に変換
            dx = dx.transpose(0, 3, 1, 2) # 軸をNCHWに入れ替え
        elif dout.ndim == 2:
            """
            画像形式以外の場合
            """
            dx = self.__backward(dout)

        return dx

    def __backward(self, dout):
        """
        ここを完成させるには、計算グラフを理解する必要があり、実装にかなり時間がかかる.
        """
        # betaの勾配
        dbeta = np.sum(dout, axis=0)
        
        # gammaの勾配(n方向に合計)
        dgamma = np.sum(self.x_std * dout, axis=0)
        
        # Xstdの勾配
        a1 = self.params["gamma"] * dout
        
        # Xmuの勾配(1つ目)
        a2 = a1 / self.std
        
        # 標準偏差の逆数の勾配(n方向に合計)
        a3 = np.sum(a1 * self.x_mu, axis=0)

        # 標準偏差の勾配
        a4 = -(a3) / (self.std * self.std)
        
        # 分散の勾配
        a5 = 0.5 * a4 / self.std
        
        # Xmuの2乗の勾配
        a6 = a5 / self.batch_size
        
        # Xmuの勾配(2つ目)
        a7 = 2.0  * self.x_mu * a6
        
        # muの勾配
        a8 = np.sum(-(a2+a7), axis=0)

        # Xの勾配
        dx = a2 + a7 +  a8 / self.batch_size # 第3項はn方向に平均
        
        self.grads["gamma"] = dgamma
        self.grads["beta"] = dbeta
        
        return dx

class Dropout(Module):
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.is_params = False

    def forward(self, x):
        if self.train:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
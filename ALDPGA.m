function [Fbest,Xbest,CNVG]=ALDPGA(N,T,lb,ub,dim,fobj)
vec_flag=[1,-1];
Nl=round((0.2*rand+0.4)*N); 
Ns=N-Nl; 
half_dim = ceil(dim/2); % 避免出现奇数维度少跑情况

% 引入的 Adam 参数
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8; %% 防止除0的小数据
% lr = 0.001; 
lr_max = 0.01;  %%学习率上限
lr_min = 0.0005; %% 学习率下限制

alpha1 = cos((1-rand(N,1))*2*pi);
m = zeros(N, dim); % N个个体的一阶矩估计
v = zeros(N, dim); % N个个体的二阶矩估计

% ------------------- 统一上下界格式 -------------------
if size(lb, 2) == 1
    lb = ones(1,dim).*lb;
    ub = ones(1,dim).*ub;
end

% ------------------- 种群初始化 -------------------
X=initialization(N,dim,ub,lb); 

fitness = zeros(1, N);
for i=1:N
    fitness(i)=feval(fobj,X(i,:));
end

[GYbest, gbest] = min(fitness);
Xbest= X(gbest,:);
Fbest=GYbest; 
CNVG = zeros(1, T); % 记录当前代的最优

% ------------------- levy飞行维度处理 ----------------
if dim <= 30
    beta = 1.5;
    scale = 0.8;
elseif dim <= 80
    beta = 1.7;
    scale = 1.2 / sqrt(dim);
else
    beta = 1.9;
    scale = 2.0 / sqrt(dim);
end

t=0; 

while t<T
    lr = lr_max - (lr_max - lr_min) * (t / T);
    %% 同步分区，因为结尾已经进行了选择和分区
    Xs=X(1:Ns,:); 
    Xl=X(Ns+1:N,:); 
    fitness_s=fitness(1:Ns);
    fitness_l=fitness(Ns+1:N);
    
    [fitnessBest_s, gbest1] = min(fitness_s);
    Xbest_s = Xs(gbest1,:);
    
    [fitnessBest_l, gbest2] = min(fitness_l);
    Xbest_l = Xl(gbest2,:);

    t=t+1; 

    %%=====光照区模拟退火 =====
    T0_light = 0.3;      % 可调：0.1~1.0
    Tf_light = 1e-5;     % 可调：1e-6~1e-4
    T_light  = T0_light * (Tf_light / T0_light)^(t / T);
    %alpha = exp(-1*t/T); % 生长限制因子
    alpha = exp(-0.5*t/T);

    % AGDO 时间相关权重 w 和自适应系数 a
    % 控制搜索的强度，代数增加，w会减小，从大范围搜索转化为逐步的精确搜索
    w =abs(rand() * ((1 / T^2) * t^2 - 2 / T * t + 1 / 2)); % 式(12)

    %% -------------------------------------------------------------------------这步与原PGA一致
    % 阴影区域边界处理
    for i=1:size(Xs,1)
        FU=Xs(i,:)>ub; FL=Xs(i,:)<lb;
        Xs(i,:)=(Xs(i,:).*(~(FU+FL)))+ub.*FU+lb.*FL;
        fitness_s(i) = fobj(Xs(i,:));
        %%写回整体
        X(i,:) = Xs(i,:);
        fitness(i) = fitness_s(i);
        if fitness_s(i)<fitnessBest_s
            fitnessBest_s=fitness_s(i);
            Xbest_s=Xs(i,:);
        end
    end
    
    % 光照区域边界处理
    for i=1:size(Xl,1)
        FU=Xl(i,:)>ub; FL=Xl(i,:)<lb;
        Xl(i,:)=(Xl(i,:).*(~(FU+FL)))+ub.*FU+lb.*FL;
        fitness_l(i) = fobj(Xl(i,:));
        %% 写回总体（注意光照区在总体里的位置）
        X(Ns+i, :) = Xl(i,:);
        fitness(Ns+i) = fitness_l(i);
        if fitness_l(i)<fitnessBest_l
            fitnessBest_l=fitness_l(i);
            Xbest_l=Xl(i,:);
        end
    end
    
    % 计算曲率因子 
    flag_index = floor(2*rand()+1); %% [0,3),floor,flag_index即1或2，概率各0.5，表示索引
    beta_curvaure=vec_flag(flag_index); %% vec_flag=[1,-1];这里的beta最终就是1*1double{1,-1};那么结果为1或者-1的概率就是各0.5
    Curvaure=beta_curvaure*(alpha-mean(fitness_s)/Fbest+ epsilon); % 根据式(17)计算曲率  
    
%% ------------------- （光照区域细胞的有丝分裂过程与梯度下降) -------------------
% ====== 光照区"自学(梯度微调)"概率，随迭代上升 ======
k_phase   = 13;        % 斜率越大，越晚切换
p_grad = @(t,T) 1./(1+exp(-k_phase*(t/T - 0.7)));   % 前期≈0，后期→1

prob_grad = p_grad(t, T);           % 随代数上升

if rand < prob_grad

Xl_original_indices = Ns + 1 : N; 

for i=1:size(Xl,1)
    X_idx = Xl_original_indices(i); % 对应原始种群X的索引

    npo_line = Xl(i,:); % 初始候选搜索线


    for k = 1:half_dim 
        % 构造一条候选"搜索线" npo_line
        if k == 1 % 表示在搜索早期，仅进行一次动量更新，追求较快的收敛
            npo_line = w .* Xl(i, :) + alpha1(i) .* Xl(i, :);
        else % 在搜索后期，通过正余弦扰动以调整方向，实现周期的扰动，避免陷入局部最优解
            if dim>=50
                % 高纬度避免过度震荡
                npo_line = Xl(i, :) + (sin(2 * pi * t /dim)) .* npo_line;
            else
                npo_line = Xl(i, :) + (sin(2 * pi * dim * t)) .* npo_line;
            end
        end
        % 只在光照区内选两个参考个体
        L = size(Xl, 1);
        idx_pool = randperm(L);
        idx_pool(idx_pool == i) = [];
        if numel(idx_pool) < 2
            break; % 光照区个体太少，跳过
        end
        a1 = idx_pool(1);
        a2 = idx_pool(2);

        % ：相对适应度方向因子 zeta（用以指示改进方向）
        % 类似 Adam 中判断梯度方向的符号，保证更新方向正确。
        zeta = ((fitness_l(a1) - fitness_l(i)) / abs(fitness_l(a1) - fitness_l(i) + epsilon));
        po_mean = mean(Xl);  %% 光照区均值
        P = (po_mean - npo_line); % P 为"偏移方向"

        % target = (Xbest_l + mean(Xl) ) / 2; %% 不仅以最优解为目标，而是考虑最优的同时也考虑均值，保持群体结构。

        [f, m(X_idx,:), v(X_idx,:)] = gtdt(Xbest, P, lr, t, beta1, beta2, epsilon, ...
            m(X_idx,:), v(X_idx,:)); % m 一阶矩阵估计，v二阶矩阵估计,也要返回,历史经验值

        a_rand = (1 - t / T) * rand(1, dim);
        % 候选解 1
        npo_1a = npo_line + zeta .* a_rand .* (f - Xl(a1, :)) - a_rand .* (npo_line - Xl(a2, :));

        % 候选解 2
        npo_1b = Xl(a1, :) + a_rand .* (f - Xl(a2, :));

        % 逐维选择（在候选解 1 和 2 之间二选一）
        for j = 1:dim
            if rand < 0.5 + 0.2*sin(2*pi*t/T) % 选择的概率平滑变化，避免维度间震荡
                npo_line(1, j) = npo_1b(1, j);
            else
                npo_line(1, j) = npo_1a(1, j);
            end
        end

        % 边界处理
        npo_line = max(npo_line, lb);
        npo_line = min(npo_line, ub);

        % 局部更新：若更优则接受
        % 计算适应度,不严格贪心选择最优解，而是采用温度衰减机制，“细胞自学”
        newfit = fobj(npo_line);
        delta = newfit - fitness(X_idx);
        if (delta < 0) || (rand < exp(-delta / max(T_light, eps)))
            % 总体中更新
            X(X_idx, :)     = npo_line;
            fitness(X_idx)  = newfit;
            % 光照子群中更新
            Xl(i,:)         = npo_line;
            fitness_l(i)    = newfit;
            % 仍保持全局最优的"贪心"更新
            if newfit < Fbest
                Fbest = newfit;
                Xbest = npo_line;
            end
        end
    end
end
%%采用原PGA细胞有丝分裂形式前期快速搜索————————————————————————
else
    Xl_original_indices = Ns + 1 : N;  % 光照区在总体中的索引
    numL = size(Xl,1); 
    Xlnew_all = zeros(numL*2, dim);    % 每个细胞产生2个子代
    fitness_lNew_all = zeros(1, numL*2);
    new_idx = 1;
    
    for i = 1:numL
        r2 = 2 * rand(1,dim) - 1;   % [-1,1)
        r3 = 2 * rand(1,dim) - 1;
        r4 = 2 * rand(1,dim) - 1;
        dd = randi([1,N]);          % 随机参考个体索引
        beta = vec_flag(randi([1,2]));  % ±1，控制方向
        
        % ---------- 变异算子 ----------
        Xlnew1(1,:) = X(dd,:) ...
            + beta * alpha .* r2 .* abs(X(dd,:) - Xl(i,:)) ...
            + beta * alpha .* r3 .* abs(Xbest_l - Xl(i,:));
        
        % ---------- 生长素重分配算子 ----------
        Xlnew1(2,:) = Xl(i,:) ...
            + alpha .* r4 .* abs(Xbest_l - Xl(i,:));
        
        % ---------- 边界检查 ----------
        for j = 1:2
            Tp = Xlnew1(j,:) > ub;
            Tm = Xlnew1(j,:) < lb;
            Xlnew1(j,:) = (Xlnew1(j,:).*(~(Tp+Tm))) + ub.*Tp + lb.*Tm;
            fitness_lNew(j) = fobj(Xlnew1(j,:));
        end
        
        % === 写入预分配矩阵 ===
        Xlnew_all(new_idx:new_idx+1,:) = Xlnew1;
        fitness_lNew_all(new_idx:new_idx+1) = fitness_lNew;
        new_idx = new_idx + 2;
    end
    
    % 合并原光照群与新子代
    Xl = [Xl; Xlnew_all];
    fitness_l = [fitness_l, fitness_lNew_all];
    % ---------- 选择最优个体 ----------
    [fitness_l, SortOrder] = sort(fitness_l);
    Xl = Xl(SortOrder,:);
    Xbest_l = Xl(1,:);
    fitnessBest_l = fitness_l(1);
    
    % 保留前 Nl 个
    Xl = Xl(1:Nl,:);
    fitness_l = fitness_l(1:Nl);
    
    % ---------- 更新全局最优 ----------
    if fitnessBest_l < Fbest
        Fbest = fitnessBest_l;
        Xbest = Xbest_l;
    end
end
%% ------------------- 阴影区活跃搜索（AGDO式全局探索） -------------------
for i = 1:size(Xs,1)
    
    X_idx = i;     % 阴影区在总体X的索引
    npo_line = Xs(i,:);

    for k = 1:half_dim
        % ：候选搜索线
        if k == 1
            npo_line = w .* Xs(i,:) + alpha1(i) .* Xs(i,:);
        else
            npo_line = Xs(i,:) + (sin(2 * pi * dim * t)) .* npo_line;
        end

        % 阴影部分也要参考光照区中和全局最优的解，向优秀解靠近
        idx_pool = randperm(N); % 全局细胞
        idx_pool(idx_pool ==X_idx) = [];
        if numel(idx_pool) < 2, break; end
        a1 = idx_pool(1); 
        a2 = idx_pool(2);

        % 相对适应度方向因子
        zeta = (fitness(a1) - fitness(i)) / (abs(fitness(a1) - fitness(i)) + epsilon);

        % 阴影区均值
        mean_s = mean(X,1);
        P = mean_s - npo_line;

        % Adam增强梯度下降
        [f_dir, m(i,:), v(i,:)] = gtdt(Xbest, P, lr, t, beta1, beta2, epsilon, ...
            m(i,:), v(i,:));

        % 自适应系数 a（随迭代线性衰减并引入随机性）
        a_rand = (1 - t / T) * rand(1, dim);

        % 候选解1、2
        npo_1a = npo_line + zeta .* a_rand .* (f_dir - X(a1,:)) ...
                          - a_rand .* (npo_line - X(a2,:));
        npo_1b = X(a1,:) + a_rand .* (f_dir - X(a2,:));

        % 二选一
        for j = 1:dim
            if rand / k > rand
                npo_line(j) = npo_1b(j);
            else
                npo_line(j) = npo_1a(j);
            end
        end

        % 边界
        npo_line = max(npo_line, lb);
        npo_line = min(npo_line, ub);

        % 若更优则接受
        newfit = fobj(npo_line);
        if newfit < fitness(X_idx)
            X(X_idx,:) = npo_line;
            fitness(X_idx) = newfit;
            Xs(i,:) = npo_line;
            fitness_s(i) = newfit;

            if newfit < Fbest
                Fbest = newfit;
                Xbest = npo_line;
            end
        end
    end
end

%% ------------------- 选择与更新 -------------------

% 统一选择最佳阴影区域细胞
[fitness_s, SortOrder]=sort(fitness_s);
Xs=Xs(SortOrder,:);
[fitnessBest_s,Sbest]=min(fitness_s);
Xbest_s=Xs(Sbest,:);
Xs=Xs(1:Ns,:);
fitness_s=fitness_s(1:Ns);

% 统一选择最佳光照区域细胞
[fitness_l, SortOrder]=sort(fitness_l);
Xl=Xl(SortOrder,:);
[fitnessBest_l,lbest]=min(fitness_l);
Xbest_l=Xl(lbest,:);
Xl=Xl(1:Nl,:);
fitness_l=fitness_l(1:Nl);

% 统一回整体
X=[Xs;Xl];
fitness=[fitness_s fitness_l];

% 统一更新全局最优解
if fitnessBest_l<Fbest
    Fbest=fitnessBest_l;
    Xbest=Xbest_l;
elseif fitnessBest_s<Fbest
    Fbest=fitnessBest_s;
    Xbest=Xbest_s;
end
    
%% ------------------- 细胞伸长阶段 -------------------
% p = 1 - exp(-5*t/T);
p = 1 ./ (1 + exp(10 * (t / T - 0.5)));  % Sigmoid 衰减函数
if rand > p
    % --------- 阶段1：Lévy 全局探索 ---------
    % 采用分段式的莱维飞行，避免Xbest自身就不是最优解，全体向错误的方向飞行。
    % ---------------------------------------
    tau = t/T;
    decay = (1-tau) ^0.7;
    Step = levy(N,dim,beta);
    span =(ub - lb );
    if tau > 0.4
       Xnew = X + scale * decay *  Step .* repmat(span,N,1);
    else
        Elite = repmat(Xbest, N, 1);
        Xnew  = X + scale * 0.8  * Step .* (Elite - X) .* rand(N, dim);
    end

    % 边界 + 接受
    Xnew = max(Xnew, lb);
    Xnew = min(Xnew, ub);

    for i = 1:N
        newfit = fobj(Xnew(i,:));
        if newfit < fitness(i)
            X(i,:) = Xnew(i,:);
            fitness(i) = newfit;
            if newfit < Fbest
                Fbest = newfit;
                Xbest = Xnew(i,:);
            end
        end
    end

else
    % --------- 阶段2：细胞伸长（后期收敛） ---------
    for i = 1:N
        r = 2*rand(1,dim)-1;
        beta = vec_flag(randi([1,2]));
        FOC = r .* (Curvaure .* (X(i,:) - Xbest));
        Cell_vicinity = beta * alpha .* r .* (X(i,:) + Xbest)/2;
        Xnew = X(i,:) + FOC + 0.3 * Cell_vicinity;
        Xnew = max(Xnew, lb);
        Xnew = min(Xnew, ub);

        newfit = fobj(Xnew);
        if newfit < fitness(i)
            X(i,:) = Xnew;
            fitness(i) = newfit;
            if newfit < Fbest
                Fbest = newfit;
                Xbest = Xnew;
            end
        end
    end
end

%% ------------------- 最终排序选择与分区更新 -------------------
    if mod(t, 20) == 0  % 每20代再做一次强排序
        [fitness, SortOrder] = sort(fitness);
        X       = X(SortOrder,:);
        m       = m(SortOrder,:);
        v       = v(SortOrder,:);
        alpha1  = alpha1(SortOrder,:);
    end
        
    if fitness(1) < Fbest
        Fbest=fitness(1);
        Xbest=X(1,:);
    end
    
    % 保留前 N 个（防止新增个体导致溢出）
    if size(X,1) > N
        X = X(1:N,:);
        fitness = fitness(1:N);
        m = m(1:N,:);
        v = v(1:N,:);
        alpha1 = alpha1(1:N,:);
    end
        
    % 更新种群分区
    Nl=round((0.2*rand+0.4)*N); 
    Ns=N-Nl; 
    
    if Ns == 0
        Ns = 1;
        Nl = N - 1;
    end
    
    Xs=X(1:Ns,:);
    Xl=X(Ns+1:N,:);
    fitness_s=fitness(1:Ns);
    fitness_l=fitness(Ns+1:N);
    
    [fitnessBest_s, gbest1] = min(fitness_s);
    Xbest_s = Xs(gbest1,:);
    [fitnessBest_l, gbest2] = min(fitness_l);
    Xbest_l = Xl(gbest2,:);
    
    
    CNVG(t)=Fbest;
end
end

% =========================================================================
% 辅助函数（子函数）
% =========================================================================

% ---------------- 辅助函数：带 Adam 与动量的"增强梯度下降"映射 ----------------
function [f, m_new, v_new] = gtdt(Xbest, P, lr, t, beta1, beta2, epsilon, m_old, v_old)    
    C = 1 - rand;
    grad = Xbest - C .* P; % 使用全局最优 Xbest 引导梯度方向
    
    % 更新一阶矩（动量）估计 (m)
    m_new = beta1 * m_old + (1 - beta1) * grad;
    % 更新二阶矩（方差）估计 (v)
    v_new = beta2 * v_old + (1 - beta2) * (grad .^ 2);
    
    % 偏差校正
    % t=0 时，1-beta1^t 和 1-beta2^t 为 0，确保 t >= 1
    m_hat = m_new / (1 - beta1 ^ t);
    v_hat = v_new / (1 - beta2 ^ t);
    
    % Adam 核心更新 (返回新位置 f)
    f = Xbest - lr * m_hat ./ (sqrt(v_hat) + epsilon);
end

% ---------------- 辅助函数：Lévy 飞行 ----------------
function [z] = levy(n, m, beta)
% Lévy 飞行步长生成器
    if nargin < 3
        beta = 1.2; 
    end
    
    num = gamma(1 + beta) * sin(pi * beta / 2);
    den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2);
    sigma_u = (num / den)^(1 / beta); 
    
    u = random('Normal', 0, sigma_u, n, m);
    v = random('Normal', 0, 1, n, m);

    % u = sigma_u .* randn(n, m);
    % v = randn(n, m);
    
    z = u ./ (abs(v).^(1 / beta));
end
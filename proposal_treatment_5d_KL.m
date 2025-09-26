close all;
clear all;
clc;

%Uの要素まで最適化する！！

%% 定義
%システム行列
A0 = [ 0.386,  0, 0,  0,  0.161;
       0,  0.461,  0, -0.047, 0;
      -0.042,  0,  0.317,  0.134, -0.117;
       0, 0,  0.134,  0.401, -0.157;
       0.161, 0, -0.117, -0.157,  0.85];%ものによってはうまくいく

n = length(A0);

%論文に乗せたやつ
% U_final = zeros(n,n);
% U_final(1,2) = -0.0001;

%response letter1
% U_final = zeros(n,n);
% U_final(1,2) = -0.0005;

%response letter2
% U_final = zeros(n,n);
% U_final(4,3) = -0.0001;

%response letter3
% U_final = zeros(n,n);
% U_final(4,4) = -0.0001;

%response letter4
U_final = zeros(n,n);

A = A0+U_final;

diag_w = [0.003 0.003 0.003 0.003 0.003];
%diag_w = [0.003 0.003];
P_w = diag(diag_w);
P_w_sqrt = sqrtm(P_w);

mu_init = 0;
Sigma_init = [7 3 0 0 0;
              3 5 0 0 0;
              0 0 6 2 0;
              0 0 2 4 1;
              0 0 0 1 3];
% Sigma_init = [7 3;
%            3 5];

%シミュレーション条件
time_init = 1;
time_step = 1;
time_term = 51;
time = time_init:time_step:time_term;
length_time = length(time);


%% データ取得
T_list = [1 51]; %スナップショットの時刻リスト
%T_list = [1 26 51];
PN = length(T_list); %スナップショットをとる時刻の数
N0 = 1000;
N = N0*PN;
N_list = zeros(1,PN+1);
N_list(1,1) = 0; %データ数のリスト
for ip = 1:PN
    N_list(1,ip+1) = N0*ip;
end 
TD_list = zeros(1,PN); %スナップショットの離散化時刻リスト
for itp = 1:PN
    TD_list(1,itp) = time_init + T_list(itp)/time_step -1;
end

%初期値の軌道
x_end_list = zeros(n,N);
Sigma_init_sqrt = sqrtm(Sigma_init);
for it = 1:N
    x0s_list = zeros(n,length_time);
    x0s_list(:,1) = mu_init*randn(n,1) + Sigma_init_sqrt*randn(n,1);
    for jt = time_init:time_term-1
        x0s_list(:,jt+1) = A0*x0s_list(:,jt) + P_w_sqrt*randn(n,1);
    end
    x_end_list(:,it) = x0s_list(:,time_term);
end
mu_x0 = mean(x_end_list,2);
Sigma_x0 = cov(x_end_list.');
Sigma_x0_sqrt = sqrtm(Sigma_x0);

%A精度向上のための繰り返し
h_iter = 10;%60;
%初期値
A_hat = eye(n); %initial value of A
K_list = []; %A0の誤差（KLダイバージェンス）
E_list = []; %A0の誤差（frobeniusノルム）
W_list = []; %ワッサースタイン距離
IJ_list = [];
fval_JP = [];

%治療の準備
% %定義
Sigmah = [ 0.0053    0        -0.0007    0         0.0022;
           0         0.0051    0         0         0     ;
          -0.0007    0         0.0052    0        -0.0021;
           0         0         0         0.0058   -0.0026;
           0.0022    0        -0.0021   -0.0026    0.0100 ];
Sigmah_sqrt = sqrtm(Sigmah);

iter_A = 20;%20;
iter_U = 10000;
eta = 0.1;
epsilon = 10^(-3); %Aの推定の終了条件
varepsilon = 10^(-4); %Uの求解の終了条件
%delta = 10^(-7);
delta = 5*10^(-8); %全体のアルゴリズムの終了条件
%目的関数
f_fun = @(P) 0.5*(trace(Sigmah\P) - n + log(det(Sigmah)/det(P)));
diff_f_P_fun = @(P) numerical_matrix_gradient(f_fun, P);

A_tilde_m1 = zeros(n,n);

for ht = 1:h_iter  
    % シミュレーション
    x_list = zeros(n,length_time*N);
    for it = 1:N
        xs_list = zeros(n,length_time);
        xs_list(:,1) = mu_x0 + Sigma_x0_sqrt*randn(n,1);
        for jt = time_init:time_term-1
            xs_list(:,jt+1) = A*xs_list(:,jt) + P_w_sqrt*randn(n,1);
        end
        x_list(:,1+length_time*(it-1):length_time*it) = xs_list;
    end
    
    % スナップショット収集
    Y_list = zeros(n,N);
    for ipy = 1:PN
        y_list = zeros(n,N0);
        for jpy = 1:N0
            y_list(:,jpy) = x_list(:,TD_list(1,ipy)+length_time*(N_list(ipy)+jpy-1));
        end
        Y_list(:,1+(ipy-1)*N0:ipy*N0) = y_list;
    end
    
    % データの期待値と共分散の計算
    muY_list = zeros(n,PN);
    for imu = 1:PN
        muY_list(:,imu) = mean(Y_list(:,1+(imu-1)*N0:imu*N0),2);
    end
    SigmaY_list = zeros(n, n, PN);
    for iSigma = 1:PN
        data_block = Y_list(:, 1+(iSigma-1)*N0 : iSigma*N0).';  % N0×n
        SigmaY_list(:, :, iSigma) = cov(data_block);
    end
    
    
    %% A行列の推定

    KL_list = [];
    Error_list = [];
    
    for i = 1:iter_A
        %xstar_all = zeros(n,length_time);
        muk_all = zeros(n, length_time-1);
        Sigmak_all = zeros(n, n, length_time-1);
        AQk_all    = zeros(n, n, length_time-1);

        for iSS = 1:PN-1
            %% expectation step with SBproblem
            interval = TD_list(iSS+1)-TD_list(iSS)+1;
            time_list = 1:interval;
        
            %% initial and desired state
            Sigma_0 = SigmaY_list(:,:,iSS);
            mu_0 = muY_list(:,iSS);
            Sigma_N = SigmaY_list(:,:,iSS+1);
            mu_N = muY_list(:,iSS+1);
        
            %% SB initial state
            Gc_N0 = zeros(n,n);
            for ig = 1:interval-1
                Gc_N0 = Gc_N0 + (A_hat\eye(n))^(ig+1)*P_w_sqrt*(P_w_sqrt.')*((A_hat\eye(n))^(ig+1)).';
            end
            Gc_N0_sqrt = sqrtm(Gc_N0); 
            Gc_N0_sqrt = real((Gc_N0_sqrt+Gc_N0_sqrt.')/2);
        
            S_0 = Gc_N0_sqrt\Sigma_0/Gc_N0_sqrt;
            S_0_sqrt = sqrtm(S_0);
            S_0_sqrt = real((S_0_sqrt+S_0_sqrt.')/2);
            S_N = Gc_N0_sqrt\((A_hat\eye(n))^interval)*Sigma_N*((A_hat\eye(n))^interval).'/Gc_N0_sqrt;
        
            Q_mk = Gc_N0_sqrt*S_0_sqrt/(S_0 + (1/2)*eye(n) - (S_0_sqrt*S_N*S_0_sqrt + (1/4)*eye(n))^(1/2))*S_0_sqrt*Gc_N0_sqrt;
        
            %Fref = S_0 + 1/2*eye(n) - sqrtm(S_0_sqrt*S_N*S_0_sqrt + 1/4*eye(n));
            %Bref = - S_0 + 1/2*eye(n) + sqrtm(S_0_sqrt*S_N*S_0_sqrt + 1/4*eye(n));
        
            %% optimal trajectories
            xstar_list = zeros(n,interval);
            xstar_list(:,1) = mu_0 + Sigma_x0_sqrt*randn(n,1);
            ukstar_list = zeros(n,interval-1);
            muk_list = zeros(n,interval);
            muk_list(:,1) = mu_0;
            Sigmak_list = zeros(n,n,interval);
            Sigmak_list(:,:,1) = Sigma_0;
            AQk_list = zeros(n,n,interval-1);
        
            for time = 1:interval-1
                Q_mk1 = A_hat*Q_mk*A_hat.' - P_w_sqrt*P_w_sqrt.';
                %Q_mk1 = Q_mk1 + 10^(-6) * eye(n);
                % eig_Q = eig(Q_mk1);
                % if any(eig_Q < 0)
                %     Q_mk1 = zeros(n);
                %     break
                % end
                A_Qk = A_hat - P_w_sqrt/(eye(n)+P_w_sqrt.'/(Q_mk1)*P_w_sqrt)*P_w_sqrt.'/(Q_mk1)*A_hat;
                B_div = sqrtm(eye(n)+P_w_sqrt.'/Q_mk1*P_w_sqrt);
                B_div = real((B_div+B_div.')/2);
                B_Qk = P_w_sqrt/B_div;
        
                if ~isreal(B_Qk)
                    warning('B_Qk has become complex at iteration %d', i);
                    break
                end
        
                AQk_list(:,:,time) = A_Qk;
                muk_list(:,time+1) = A_Qk*muk_list(:,time);
                Sigmak_list(:,:,time+1) = A_Qk*Sigmak_list(:,:,time)*A_Qk.' + B_Qk*B_Qk.';
                      
                xstar_list(:,time+1) = A_Qk*xstar_list(:,time) + B_Qk*randn(n,1);
        
                Q_mk = Q_mk1;
            end
        
            muk_all(:,TD_list(iSS):TD_list(iSS+1)) = muk_list;
            for t = 1:interval
                Sigmak_all(:,:,TD_list(iSS) + t - 1) = Sigmak_list(:,:,t);
            end
            for t = 1:interval-1
                AQk_all(:,:,TD_list(iSS) + t - 1) = AQk_list(:,:,t);
            end
        end


        if ~isreal(B_Qk)
           warning('B_Qk has become complex at iteration %d', i);
           break
        end

        E_nm = zeros(n, n);
        for inm = 1:length_time-1
            E_nm = E_nm + AQk_all(:,:,inm) * ...
                (Sigmak_all(:,:,inm) + muk_all(:,inm) * muk_all(:,inm).');
        end
        
        E_mm = zeros(n, n);
        for imm = 1:length_time-1
            E_mm = E_mm + Sigmak_all(:,:,imm) + ...
                muk_all(:,imm) * muk_all(:,imm).';
        end
       

        %% maximization step
        A_hat = E_nm/E_mm;

        A_tilde = ((ht-1)/ht)*A_tilde_m1 + (1/ht)*(A_hat - U_final);
        
        A_hat = A_tilde + U_final


        %frobeniusノルム
        Error = norm(A_hat-A,'fro')/norm(A,'fro');
        Error_list(1,i) = Error;

        %KL
        x_KL_list = zeros(n,length_time*N);
        for itKL = 1:N
            xs_KL_list = zeros(n,length_time);
            xs_KL_list(:,1) = mu_x0 + Sigma_x0_sqrt*randn(n,1);
            for jtKL = time_init:time_term-1
                xs_KL_list(:,jtKL+1) = A_hat*xs_KL_list(:,jtKL) + P_w_sqrt*randn(n,1);
            end
            x_KL_list(:,1+length_time*(itKL-1):length_time*itKL) = xs_KL_list;
        end
        Y_KL_list = zeros(n, PN * N0);
        for ipyKL = 1:PN
            y_KL_list = zeros(n, N0);
            for jpyKL = 1:N0
                idx = TD_list(1, ipyKL) + length_time * (N_list(ipyKL) + jpyKL - 1);
                y_KL_list(:, jpyKL) = x_KL_list(:, idx);
            end
            Y_KL_list(:, (ipyKL-1)*N0 + 1 : ipyKL*N0) = y_KL_list;
        end
        
        muY_KL_list = zeros(n, PN);
        for imuKL = 1:PN
            muY_KL_list(:, imuKL) = mean(Y_KL_list(:, (imuKL-1)*N0 + 1 : imuKL*N0), 2);
        end
        
        SigmaY_KL_list = zeros(n, n * PN);
        for iSigmaKL = 1:PN
            data = Y_KL_list(:, (iSigmaKL-1)*N0 + 1 : iSigmaKL*N0);
            SigmaY_KL_list(:, (iSigmaKL-1)*n + 1 : iSigmaKL*n) = cov(data.');
        end
        
        %KLで収束性の確認
        KL = 0;
        for iKL = 1:PN
            mu_KL = muY_KL_list(:, iKL);
            Sigma_KL = SigmaY_KL_list(:, (iKL-1)*n + 1 : iKL*n);
            mu_data = muY_list(:, iKL);
            Sigma_data = SigmaY_list(:, (iKL-1)*n + 1 : iKL*n);
        
            kl = (1/2) * (log(det(Sigma_data) / det(Sigma_KL)) ...
                - trace(eye(n)) + trace(Sigma_data \ Sigma_KL) ...
                + (mu_data - mu_KL).' / Sigma_data * (mu_data - mu_KL));
            KL = KL + kl;
        end

        KL_list(1,i) = KL;
    
        if i > 1
            if norm(KL_list(i) - KL_list(i-1)) < epsilon
            %if KL_list(i) > KL_list(i-1)
                break
            end
        end           
    end
    

    figure(1)
    hold on
    plot(1:i,Error_list)
    xlabel('iteration','fontsize', 18) 
    ylabel('A-A_0','fontsize', 18) 
    %title('Aの誤差の推移')
    E_list(1,ht) = Error_list(1,i);
     
 
    
    %% 治療    
    %Uの求解
    % 並列処理用にインデックスを1～n^2のベクトルに変換
    idx_list = 1:n^2;
    
    % 出力変数の事前確保（並列ループ内で使う場合は注意が必要）
    best_fval_cell = cell(n^2, 1);
    best_c_list_cell = cell(n^2, 1);
    best_fval_list_cell = cell(n^2, 1);
    best_ij_cell = cell(n^2, 1);
    best_u_val_cell = zeros(n^2, 1);
    
    parfor idx = 1:n^2
        ui = ceil(idx / n);
        uj = mod(idx-1, n) + 1;
    
        % 変数初期化
        c_init = -0.4;
        c_val = c_init;
        c_list_i = zeros(1, iter_U);
        fval_list_i = zeros(1, iter_U);
        best_fval_local = Inf;
        best_c_list_local = [];
        best_fval_list_local = [];
        best_u_val_local = 0;

        valid_len = iter_U;
    
        for i_U = 1:iter_U
            U = zeros(n,n);
            U(ui, uj) = c_val;
    
            A_U = A_tilde + U;
            Sigma = dlyap(A_U, P_w);

            % Sigmaが正定値かチェック
            [~, p] = chol(Sigma);
            if p ~= 0
                % cholが失敗した場合 (pが0でない)、Sigmaは正定値ではない
                % この候補は無効とし、ループを抜ける
                fval_list_i(i_U) = Inf; % コストに無限大を代入
                c_list_i = c_list_i(1:i_U);
                fval_list_i = fval_list_i(1:i_U);
                break;
            end
    
            diffU = zeros(n,n);
            diffU(ui, uj) = 1;
    
            dpar = diffU * Sigma * A_U' + A_U * Sigma * diffU';
            gradSigma = dlyap(A_U, dpar);
    
            diff_f_matrix = diff_f_P_fun(Sigma);
            gradf = sum(diff_f_matrix .* gradSigma, 'all');
    
            c_val = c_val - eta * gradf;
            c_list_i(i_U) = c_val;
            fval_list_i(i_U) = f_fun(Sigma);
    
            if abs(gradf) < varepsilon
                c_list_i = c_list_i(1:i_U);
                fval_list_i = fval_list_i(1:i_U);
                break
            end
        end
    
        final_fval = fval_list_i(end);
        if final_fval < best_fval_local
            best_fval_local = final_fval;
            best_c_list_local = c_list_i;
            best_fval_list_local = fval_list_i;
            best_u_val_local = c_val;
        end
    
        % parfor内でセルに保存
        best_fval_cell{idx} = best_fval_local;
        best_c_list_cell{idx} = best_c_list_local;
        best_fval_list_cell{idx} = best_fval_list_local;
        best_ij_cell{idx} = [ui, uj];
        best_u_val_cell(idx) = best_u_val_local;
    end
    
    % parfor終了後に結果の統合
    [best_fval, idx_best] = min(cell2mat(best_fval_cell));
    best_ij = best_ij_cell{idx_best};
    best_c_list = best_c_list_cell{idx_best};
    best_fval_list = best_fval_list_cell{idx_best};
    best_u_val = best_u_val_cell(idx_best);

    %最適なUとA0において，J_Pの値を確認
    U_JP = zeros(n,n);
    U_JP(best_ij(1,1),best_ij(1,2)) = best_u_val;
    A_U_JP = A0 + U_JP;
    Sigma_JP = dlyap(A_U_JP, P_w);
    fval_JP(ht) = f_fun(Sigma_JP);
    

    figure(2)
    hold on
    plot(best_fval_list)
    xlabel('iteration','fontsize', 18) 
    ylabel('J_P','fontsize', 18) 
    %title('Aの誤差の推移')

    % 最適な結果を代入
    c_list = [];
    fval_list = [];
    ij_list = [];
    c_list = best_c_list;
    fval_list = best_fval_list;
    ij_list = best_ij;
    IJ_list(:,ht) = ij_list;

    
    %推定結果
    U_final = zeros(n,n);
    U_final(ij_list(1,1),ij_list(1,2)) = best_u_val;

    W_list(1,ht) = fval_list(1,length(fval_list));
    if ht > 1
        if norm(W_list(ht) - W_list(ht-1)) < delta
            break
        end
    end
    
    A = A0+U_final; %データ収集用のA行列
    A_hat = A_tilde + U_final; %A行列推定の際の初期値
    A_tilde_m1 = A_tilde; %1ステップ前のA_tildeの情報

    
end
hold off

Sigma_JP = dlyap(A0, P_w);
fval_JP = [f_fun(Sigma_JP) fval_JP];


%% 結果の確認

%%散布図での確認
%データ取得
num= 1000;
x_before_scatter = zeros(n,num);
x_after_scatter = zeros(n,num);
for k = 1:num
    %治療前のデータ
    x_before = zeros(n,length_time);
    x_before(:,1) = mu_x0 + Sigma_x0_sqrt*randn(n,1);
    for jt = time_init:time_term-1
        x_before(:,jt+1) = A0*x_before(:,jt)+P_w_sqrt*randn(n,1);
    end
    x_before_scatter(:,k) = x_before(:,time_term);
    %治療後のデータ
    x_after = zeros(n,length_time);
    x_after(:,1) = mu_x0 + Sigma_x0_sqrt*randn(n,1);
    for jt = time_init:time_term-1
        x_after(:,jt+1) = A*x_after(:,jt)+P_w_sqrt*randn(n,1);
    end
    x_after_scatter(:,k) = x_after(:,time_term);
end

% 主成分の取得
[V, D_eig] = eig(Sigmah);
[eigvals, idx_PCA] = sort(diag(D_eig), 'descend');
eigvecs = V(:, idx_PCA);           % 各列が主成分方向
% 射影行列（PC1, PC2）
W = eigvecs(:, 1:2);           % 5×2（第1・第2主成分）
% 射影後の2次元共分散行列（Sigma in PC space）
Sigmah_2D = W' * Sigmah * W;
% 99%信頼楕円を描画するためのスケーリング（カイ二乗分布 2自由度で99%）
s = sqrt(chi2inv(0.99, 2));  % ≈ 3.0349
% 楕円の点を生成
theta = linspace(0, 2*pi, 200);
circle = [cos(theta); sin(theta)];  % 単位円
% 共分散楕円に変換
[U_2D, S_2D, ~] = svd(Sigmah_2D);  % 特異値分解で回転とスケーリング
ellipse = U_2D * sqrt(S_2D) * circle * s;

%制御前データの処理
% データを主成分空間に射影（センタリング付き）
x_before_centered = x_before_scatter - mean(x_before_scatter, 2);
x_before_proj = W' * x_before_centered;          % 2×50
% 散布図と共分散楕円をプロット
figure(3)
hold on;
scatter(x_before_proj(1,:), x_before_proj(2,:), 50, 'filled','MarkerFaceAlpha', 0.6);
plot(ellipse(1,:), ellipse(2,:), 'r', 'LineWidth', 2);
xlabel('PC1','fontsize', 18);
ylabel('PC2','fontsize', 18);
%title('Projection of Data Before Control onto PC1-PC2 with 99% Confidence Ellipse');
% 軸の範囲を指定
xlim([-1.2, 1.2]);
ylim([-0.8, 0.8]);
axis equal;
grid on;
legend('Projected Data without Control', 'Covariance Ellipse of $\Sigma_{\mathrm{ref}}$', 'Location', 'best','Interpreter', 'latex');

%制御後データの処理
% データを主成分空間に射影（センタリング付き）
x_after_centered = x_after_scatter - mean(x_after_scatter, 2);
x_after_proj = W' * x_after_centered;          % 2×50
% 散布図と共分散楕円をプロット
figure(4)
hold on;
scatter(x_after_proj(1,:), x_after_proj(2,:), 50, 'filled','MarkerFaceAlpha', 0.6);
plot(ellipse(1,:), ellipse(2,:), 'r', 'LineWidth', 2);
xlabel('PC1','fontsize', 18);
ylabel('PC2','fontsize', 18);
%title('Projection of Data After Control onto PC1-PC2 with 99% Confidence Ellipse');
xlim([-1.2, 1.2]);
ylim([-0.8, 0.8]);
axis equal;
grid on;
legend('Projected Data with Control', 'Covariance Ellipse of $\Sigma_{\mathrm{ref}}$', 'Location', 'best','Interpreter', 'latex');



figure(6)
plot(1:length(c_list),c_list,LineWidth=1.5)
xlabel('Iterations of the CS Algorithm','fontsize', 18) 
ylabel('$u$','Interpreter', 'latex','fontsize', 18) 
%title('uの値の推移')

figure(7)
plot(1:length(fval_list),fval_list,LineWidth=1.5)
xlabel('Iterations of the CS Algorithm','fontsize', 18) 
ylabel('$J_P$','Interpreter', 'latex','fontsize', 18) 
%title('目的関数の推移')


figure(8)
plot(1:length(W_list),W_list,LineWidth=1.5)
xlabel('iteration','fontsize', 18) 
ylabel('$J_P$','Interpreter', 'latex','fontsize', 18)
%title('収束後の目的関数の推移')

figure(9)
plot(0:length(fval_JP)-1,fval_JP,LineWidth=1.5)
xlabel('Iterations of the D-CS Algorithm','fontsize', 18) 
ylabel('$J_P$','Interpreter', 'latex','fontsize', 18)
%title('収束後の目的関数の推移')

figure(11)
plot(1:length(E_list),E_list,LineWidth=1.5)
xlabel('Iterations of the D-CS Algorithm','fontsize', 18) 
ylabel('$\Delta A$','Interpreter', 'latex','fontsize', 18) 
% title('Aの誤差の推移')

figure(12)
hold on
plot(1:length(IJ_list),IJ_list(1,:),LineWidth=1.5)
plot(1:length(IJ_list),IJ_list(2,:),LineWidth=1.5)
xlabel('Iterations of the D-CS Algorithm','fontsize', 18) 
ylabel('Index $(i,j)$','Interpreter', 'latex','fontsize', 18) 
legend('$i$','$j$','Interpreter', 'latex')
hold off

vecA0 = reshape(A0,1,[]);
vecA_tilde = reshape(A_tilde,1,[]);
figure(13)
scatter(vecA0,vecA_tilde,"filled")
hold on;

% --- ★★★ここから再変更★★★ ---
% 全てのTrue A0 elementsとEstimated A tilde elementsの最小値と最大値を計算
min_val = min([vecA0, vecA_tilde]);
max_val = max([vecA0, vecA_tilde]);
plot_lims = [min_val, max_val];

% plotコマンドでy=xの直線を描画
plot(plot_lims, plot_lims, 'r-', 'LineWidth', 1.5);
% --- ★★★ここまで再変更★★★ ---

hold off;
grid on
xlabel('True A0 elements','fontsize', 14)
ylabel('Estimated A tilde elements','fontsize', 14)
legend('Estimation Results', 'Perfect Estimation','Location','best')


function grad = numerical_matrix_gradient(f, P)
    n = size(P, 1);
    h = 1e-6;
    
    % 1次元化して並列ループ
    num_elements = n^2;
    grad_vec = zeros(num_elements, 1);
    
    parfor k = 1:num_elements
        [i, j] = ind2sub([n, n], k);
        
        P_plus = P;
        P_minus = P;
        P_plus(i, j) = P_plus(i, j) + h;
        P_minus(i, j) = P_minus(i, j) - h;
        
        grad_vec(k) = (f(P_plus) - f(P_minus)) / (2 * h);
    end
    
    grad = reshape(grad_vec, [n, n]);
end
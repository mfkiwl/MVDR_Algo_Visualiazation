    %%%%%%%%%%%%%%%%%%%%%%%% Simple beamforming 
    clear;clc;close all;
    
    ele=10; %阵元个数
    
    snr=10;
        
        
    snap=500;%% 快拍数,也就是信号长度
    theta0= 30; %可以改变
    theta1=-20; %可以增加或减少
    
    s1=10^(snr/20)*steering_vector_ula(theta0,ele)*randn_complex(1,snap);
    
    s2=10*steering_vector_ula(theta1,ele)*randn_complex(1,snap);
    
    n=randn_complex(ele,snap);
    
    x=s1+s2+n;
    
    r=x*x'/snap;  %自相关矩阵randn_complex
    
    w20=inv(r)*steering_vector_ula(theta0,ele); %MVDR权值计算,忽略了常数
       
       
    %     axis([20 500 -15 15])
    figure
    theta=-90:90;
    for i=1:181
        
        p20(i)=w20'*steering_vector_ula(theta(i),ele);
        
    end
    plot(theta,20*log10(abs(p20)/max(abs(p20))),'-.'), hold on
    plot(theta0,[-45:0],'--'),hold on
    plot(theta1,[-45:0],'--'),hold on
    legend('MVDR');
    xlabel('角度（度）')
    ylabel('方向图(dB)')
    xlabel('Degree(\circ)')
    ylabel('Beampatter(dB)')
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
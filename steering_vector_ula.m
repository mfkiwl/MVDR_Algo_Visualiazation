function[A]=steering_vector_ula(theta,element_num)
% This function generate steer vector
A=zeros(element_num,length(theta));
d_lamda=1/2;
source_num=1;
for i=1:source_num
    A(:,i)=exp(j*2*pi*d_lamda*sin(theta(i)*pi/180)*(0:element_num-1)');
end


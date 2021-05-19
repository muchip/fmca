clear all;
close all;
Img = imread('NCC1701D.jpg');
figure(1)
imagesc(Img);
%Rchan = double(Img(:,:,1)); 
%Gchan = double(Img(:,:,2));
%Bchan = double(Img(:,:,3)); 
%save ('Rchan.txt', 'Rchan', '-ascii');
%save ('Gchan.txt', 'Gchan', '-ascii');
%save ('Bchan.txt', 'Bchan', '-ascii');
Rchannel;
Gchannel;
Bchannel;
Img2 = Img;
Img2(:,:,1) = R;
Img2(:,:,2) = G;
Img2(:,:,3) = B;

figure(2)

imagesc(Img2)

figure(3)
surf(Img2(:,:,1)-Img(:,:,1))
shading interp
view(0,90)
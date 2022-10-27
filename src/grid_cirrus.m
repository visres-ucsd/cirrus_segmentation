xCenter=101;
yCenter=101;

se3 = strel('disk', 1);
ss=8;

M=-1*pi/2:(2*pi)/ss:1.5*pi;
K=[35 80 ];
H=1:ss*length(K);
ind=0;

Mask2=zeros(200,200);

start = 0;
for k=1:length(K)
    for i=1:ss


        ind=ind+1;



        theta = M(i) : 0.0001 : M(i+1);
        radius = K(k);
        x = radius * cos(theta) + xCenter;
        y = radius * sin(theta) + yCenter;



        x = [xCenter,  x, xCenter];
        y = [yCenter,  y, yCenter];

        maski = H(ind)*poly2mask(x,y, 200,200 );
        maski(Mask2>0)=0;
        maski = imopen(maski, se3);

        Mask2=maski+Mask2;


        figure(100);imagesc(Mask2);
        pause(0.01)
    end
end
HE_mask = Mask2(1:10:end,:)';

writematrix(Mask2, 'sector_masks.csv');

addpath(genpath('../mutils/My/'));
addpath(genpath('../ptv'));

use_refinement = 0;
resize = 0;
fast_lcc = 1;
load('4dlung-floder.mat')

for idx = 1:80  %1:10--9:80
    time_path = ['E:\4D-Lung-mat','\',time_dir(idx).name,'\'];
    load([time_path,'phase_0.mat'])
    cttest = ct;
    Min = 600;
    Max = 900;
    cttest(ct>Max)=Max;
    cttest(ct<Min)=Min;
    cttest = (cttest-Min)/(Max-Min);

    x = sum(cttest,[2,3]);
    x = squeeze(x);
    y = sum(cttest,[1,3]);
    y = squeeze(y);
    z = sum(cttest,[1,2]);
    z = squeeze(z);

    s = size(cttest);
    x0 = max(1,(find(x>max(x)*0.3,1)-30));
    l = find(x<max(x)*0.2,x0+30+2);
    x1 = min(s(1),l(end));

    y0 = max(1,(find(y>max(y)*0.4,1)-10));
    l = find(y<max(y)*0.4,y0+10+2);
    y1 = min(s(2),l(end)+8);
    
    volfix = ct;
    s = size(volfix);
    if s(3)<100
        z0 = 1;
        z1 = s(3);
    else
        z0 = ceil(s(3)/2) - 49;
        z1 = ceil(s(3)/2) + 50;
    end

    crop = [x0,x1;y0,y1;z0,z1];

    volfix = img_thr(volfix, 80, 900, 1);
    volfix = crop_data(volfix, crop);
    for mov =  [1 2 3 4 5 6 7 8 9] %1 2 3 4 5 6 7 8 9
        load([time_path,'phase_',num2str(mov),'.mat'])
        volmov = ct;
        spc = [1 1 3];
        volmov = img_thr(volmov, 80, 900, 1);
       
        % crop images 
        volmov = crop_data(volmov, crop);
        
        % configure registration
        opts = [];
        opts.loc_cc_approximate = fast_lcc;

        opts.grid_spacing = [4, 4, 3]*2;  % grid spacing in pixels
        opts.cp_refinements = 1;

        opts.display = 'off';
        opts.k_down = 0.7;
        opts.interp_type = 0;
        opts.metric = 'loc_cc_fftn_gpu';
        opts.metric_param = [1,1,1] * 2.1;
        opts.scale_metric_param = true;
        opts.isoTV = 0.11;
        opts.csqrt = 5e-3;
        opts.spline_order = 1;
        opts.border_mask = 5;
        opts.max_iters =  80;
        opts.check_gradients = 100*0;
        opts.pix_resolution = spc;
        [voldef, Tptv, Kptv] = ptv_register(volmov, volfix, opts);

        save([time_path,'dvf_0to',num2str(mov),'.mat'], 'Tptv','crop');
    end
end

clear %all
close all
clc

global Input_path
global Output_path
global Figure_path
global emg_name
global patient_ID

%%

addpath(genpath('C:\Users\Acer\Desktop\Pour alice\'))

patient_ID = 'NM005_2_smooth';
path = ['C:\Users\Acer\Desktop\Pour alice\' patient_ID];

%emg_name_002={'FCR','APL','The','FD2','FD34','FCU','ED','BR','BB','TB','DELant','DELmed','DELpost','PM','Rh'};
%emg_name_004={'FCR','FCU','ED','FD34','Brach','BB','TB','DELant','DELmed','DELpost','BR','PM','Supra','Infra','Rh'};
%emg_name_005_1={'The','FD34','BR','ECR','DELant','DELmed','DELpost','Supra','ED','BB','TB','Infra','PM','LevScap','Tmaj'};
emg_name={'ECR','BR','FD34','ED','DELant','DELmed','DELpost','BB','TB','PM','LevScap','FCU','sync'};

MVC = 0;  % 1 if MVC, 0 if not
max_pourcent = 0.8;

emg_treatment_pipeline={'Remove NaN',0,0;
    'Resample',0,0;
    'Notch',50,0;
    'High_pass_butter',4,2;
    'Low_pass_butter',4,40;
    'Rectification','median',0;
    'RMS',0.3,0};  % 'Blanking',0,0  INTERPOLATION FIRST

Input_path = [path '\Inputs'];
Output_path = [path '\Outputs'];
Figure_path = [path '\Figures'];
Data_path = [path '\Data'];
mkdir2([Input_path '\treated_emg'])
mkdir2([Input_path '\norm_max_emg']);
if MVC == 1
    mkdir2([Input_path '\norm_mvc_emg']);
end
mkdir2(Figure_path)
mkdir2([Figure_path '\treated_emg'])
mkdir2([Figure_path '\norm_emg'])
mkdir2(Data_path)
mkdir2([Data_path '\norm_max_emg'])
if MVC == 1
    mkdir2([Data_path '\norm_mvc_emg'])
end

if isempty(dir(Input_path))
    error('Input path does not exist')
end

base_on = load([Input_path '\baseline_on.mat']);
base_on = base_on.baseline_on;
base_off = load([Input_path '\baseline_off.mat']);
base_off = base_off.baseline_off;
stim_on = load([Input_path '\ONSETS.mat']);
stim_on = stim_on.ONSETS;
stim_off = load([Input_path '\OFFSETS.mat']);
stim_off = stim_off.OFFSETS;

DIR = dir2(Input_path);

max_emg = zeros(1, length(emg_name));
shift = 1e-04;

for k = 1 : length(DIR)
    if strcmp(DIR(k).name,'MVC')
        DIR2 = dir2([DIR(k).folder '\' DIR(k).name]);
        emg = load([DIR2(1).folder '\' DIR2(1).name]);
        emg = EMG_treatment(emg,emg_treatment_pipeline);
        ref = find_MVC_values(emg);

    elseif strcmp(DIR(k).name,'EMG')
        DIR2 = dir2([DIR(k).folder '\' DIR(k).name]);
        for k2=1:length(DIR2)
            emg = load([DIR2(k2).folder '\' DIR2(k2).name]);
            emg = EMG_treatment(emg, emg_treatment_pipeline);
            save([Input_path '\treated_emg\' DIR2(k2).name], 'emg', '-v7.3')
	        id = find(contains(emg.Channels,'EMG'));
            close all
            figure2()
            hold on
	        for k3=1:length(id)
	            if max(emg.Data{id(k3)}) > max_emg(k3)
		            max_emg(k3) = max(emg.Data{id(k3)});
                end
                plot(emg.Time{id(k3)},k3*shift+emg.Data{id(k3)}) 
            end
            hold off
            saveas(gcf, [Figure_path '\treated_emg\' DIR2(k2).name '.jpg'])
        end
    end
end

% Normalisation by max or MVC
DIR = dir2([Input_path '\treated_emg']);
for k = 1 : length(DIR)
    emg = load([DIR(k).folder '\' DIR(k).name]);
    emg = emg.emg;
    id = find(contains(emg.Channels,'EMG'));
    for k3=1:length(id)
	    if max(emg.Data{id(k3)}) > max_emg(k3)
		    max_emg(k3) = max(emg.Data{id(k3)});
        end
    end
end
for k = 1 : length(DIR)
    emg = load([DIR(k).folder '\' DIR(k).name]);
    emg = emg.emg;
    if MVC == 1
    	emg2 = emg;
    end
    nt = length(emg.Time{1});
    nm = length(emg_name);
    norm_max_data = -1*ones(nt, nm);
    if MVC == 1
    	norm_mvc_data = -1*ones(nt, nm);
    end
    time = -1*ones(nt, nm);
    id = find(contains(emg.Channels,'EMG'));
    close all
    figure2()
    hold on
    for k2=1:length(id)
	    emg.Data{id(k2)} = max_pourcent*emg.Data{id(k2)}/max_emg(k2); 
        norm_max_data(1:length(emg.Data{id(k2)}), k2) = max_pourcent*emg.Data{id(k2)}/max_emg(k2); 
        if MVC == 1
	        emg2.Data{id(k2)} = emg2.Data{id(k2)}/ref(id(k2));
    	    norm_mvc_data(1:length(emg2.Data{id(k2)}), k2) = emg2.Data{id(k2)}/ref(id(k2));
        end
	    time(1:length(emg.Data{id(k2)}), k2) = emg.Time{id(k2)};
        plot(emg.Time{id(k2)},k2+emg.Data{id(k2)}) 
    end
    hold off
    saveas(gcf, [Figure_path '\norm_emg\max_' DIR(k).name '.jpg'])
    save([Input_path '\norm_max_emg\' DIR(k).name], 'emg', '-v7.3')
    mkdir2([Data_path '\norm_max_emg\' DIR(k).name]);
    writematrix(norm_max_data, [Data_path '\norm_max_emg\' DIR(k).name '\EMG_norm_max_data.txt'], 'Delimiter', 'tab')
    if MVC == 1
        mkdir2([Data_path '\norm_mvc_emg\' DIR(k).name]);
        save([Input_path '\norm_mvc_emg\' DIR(k).name], 'emg2', '-v7.3')
   	writematrix(norm_mvc_data, [Data_path '\norm_mvc_emg\' DIR(k).name '\EMG_norm_mvc_data.txt'], 'Delimiter', 'tab')
    end
    writematrix(time, [Data_path '\norm_max_emg\' DIR(k).name '\EMG_time.txt'], 'Delimiter', 'tab')
end

% Stim by stim
DIR = dir2([Input_path '\norm_max_emg']);

for k = 1 : length(DIR)
    name = DIR(k).name;
    emg = load([DIR(k).folder '\' DIR(k).name]);
    emg = emg.emg;
    nt = length(emg.Time{1});
    nm = length(emg_name);
    id = -1;
    for i = 1:length(base_on)
        if strcmp(base_on{1,i}.name, name)
            id = i;
        end
    end
    base_on_i = base_on{1,id};
    stims = size(base_on_i.val);
    stims = stims(2);
    time = emg.Time{1,1};
    for k2=1:length(emg.Data)
        if contains(emg.Channels{k2},'EMG')
            emg.Data{k2} = interp1(emg.Time{k2}, emg.Data{k2}, time);
            emg.Time{k2} = time;
        end 
    end 
    for s=1:stims
        display(s)
   	    ts = stim_on{1,id};
        ts = ts.val;
        ts = ts(:,s);
        tsf = stim_off{1,id};
        tsf = tsf.val;
        tsf = tsf(:,s);
        tb = base_on_i.val;
        tb = tb(:,s);
        base_off_i = base_off{1,id};
        tbf = base_off_i.val;
        tbf = tbf(:,s);
        is = find(time > min(ts), 1, 'first'); 
        isf = find(time < max(tsf), 1, 'last'); 
        ib = zeros(nm,1);
        ibf = zeros(nm,1);
        for m=1:nm
            ib(m) = find(time > tb(m), 1, 'first'); 
            ibf(m) = find(time < tbf(m), 1, 'last');
        end 
        stim_norm_data = zeros(isf-is, nm);
        t = zeros(isf-is, nm);
        c = 0;
        for k2=1:length(emg.Data)
            if contains(emg.Channels{k2},'EMG')
                c=c+1;
        	base = emg.Data{k2};
        	base = base(ib(c):ibf(c));
        	stim = emg.Data{k2};
        	stim = stim(is:isf);
        	stim_norm_data(1:isf-is+1, abs(c)) = abs(stim-mean(base)); %/abs(max(emg.Data{k2})-mean(base));
        	t(1:isf-is+1, abs(c)) = emg.Time{k2}(is:isf);
    	    end
        end
        dir_name = [Data_path '\norm_max_emg\' DIR(k).name '\stim_' int2str(s)];
        mkdir2(dir_name);
   	    writematrix(stim_norm_data, [dir_name '\EMG_norm_data.txt'], 'Delimiter', 'tab')
   	    writematrix(t, [dir_name '\EMG_time.txt'], 'Delimiter', 'tab')
    end
end

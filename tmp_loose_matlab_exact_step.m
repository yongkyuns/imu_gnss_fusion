function tmp_loose_matlab_exact_step()
addpath(genpath('/Users/ykshin/Dev/me/open-aided-navigation/lib'));
addpath(genpath('/Users/ykshin/Dev/me/open-aided-navigation/demo/insGnssLoose'));

collectRows = evalin('base', 'exist(''outPath'',''var'')');
if collectRows
    outPath = evalin('base', 'outPath');
    if evalin('base', 'exist(''diagUntilS'',''var'')')
        diagUntilS = evalin('base', 'diagUntilS');
    else
        diagUntilS = 60.0;
    end
    rows = struct('time_s', {}, 'ttag_us', {}, 'applied_obs_types', {});
else
    if evalin('base', 'exist(''targetTime'',''var'')')
        targetTime = evalin('base', 'targetTime');
    else
        targetTime = 0.980016;
    end
end

init = jsondecode(fileread('/tmp/loose_long_init.json'));
inputDir = '/Users/ykshin/Dev/me/open-aided-navigation/data/nsr/drive/20201104_091730';

gyroData = parse_gyro_data(inputDir);
accelData = parse_accel_data(inputDir);
gnssData = parse_gnss_data(inputDir);

sMap = StateMapInsGnssLoose;
esMap = ErrorStateMapInsGnssLoose;
conf = ParamsInsGnssFilterLoose;

x = zeros(sMap.LEN, 1);
x(sMap.POS_E) = init.pos_ecef_m(:);
x(sMap.V_E) = init.vel_ecef_mps(:);
x(sMap.Q_ES) = init.q_bn(:);
x(sMap.B_W) = init.gyro_bias_radps(:);
x(sMap.B_F) = init.accel_bias_mps2(:);
x(sMap.S_W) = init.gyro_scale(:);
x(sMap.S_F) = init.accel_scale(:);
x(sMap.Q_CS) = init.q_cs(:);
P = double(init.p_full);

filTtag = uint64(init.start_ttag_us);
lastUsedGnssTtag = uint64(0);
measDb = InsGnssLooseMeasDb(16);

gyroTtags = zeros(1, length(gyroData), 'uint64');
accelTtags = zeros(1, length(accelData), 'uint64');
gnssTtags = zeros(1, length(gnssData), 'uint64');
for i = 1:length(gyroData)
    gyroTtags(i) = gyroData(i).ttag;
end
for i = 1:length(accelData)
    accelTtags(i) = accelData(i).ttag;
end
for i = 1:length(gnssData)
    gnssTtags(i) = gnssData(i).ttag;
end
allTtags = [gyroTtags, accelTtags, gnssTtags];
allEvents = [repmat(uint8(1), 1, length(gyroTtags)), repmat(uint8(2), 1, length(accelTtags)), repmat(uint8(3), 1, length(gnssTtags))];
[~, order] = sort(allTtags);

nextGyro = 1;
nextAccel = 1;
nextGnss = 1;

for i = 1:length(order)
    switch allEvents(order(i))
        case 1
            measDb = measDb.addData(gyroData(nextGyro), SensorType.GYRO);
            currGyroInd = nextGyro;
            nextGyro = nextGyro + 1;

            if currGyroInd == 1 || gyroData(currGyroInd).ttag < filTtag
                continue;
            end
            prevGyro = gyroData(currGyroInd - 1);
            if prevGyro.ttag < filTtag
                continue;
            end

            [xPred, ok, ttagNew] = propInsGnssLooseState(x, sMap, measDb);
            if ~ok
                error('propInsGnssLooseState failed');
            end
            [Phi, ok] = insGnssLooseTransMat(xPred, sMap, esMap, filTtag, ttagNew, measDb);
            if ~ok
                error('insGnssLooseTransMat failed');
            end
            [Q, ok] = insGnssLooseSysNoiseMat(xPred, sMap, esMap, filTtag, ttagNew);
            if ~ok
                error('insGnssLooseSysNoiseMat failed');
            end
            P = Phi * P * Phi' + Q;
            x = xPred;
            filTtag = ttagNew;

            t = 1e-6 * double(filTtag - uint64(init.start_ttag_us));
            [obsDb, H, gpsResidual, gpsVariance] = prep_obs(x, P, filTtag, lastUsedGnssTtag, measDb, conf, sMap, esMap);
            if collectRows
                if t > diagUntilS
                    fid = fopen(outPath, 'w');
                    fprintf(fid, '%s', jsonencode(rows));
                    fclose(fid);
                    return;
                end
                if obsDb.obsCount > 0
                    rawObsTypes = zeros(1, obsDb.obsCount);
                    for j = 1:obsDb.obsCount
                        rawObsTypes(j) = double(obsDb.obs(j).type);
                    end
                    rows(end + 1) = struct( ...
                        'time_s', t, ...
                        'ttag_us', double(filTtag), ...
                        'applied_obs_types', canonical_obs_types(rawObsTypes));
                end
            elseif abs(t - targetTime) < 1e-9
                obsTypes = zeros(1, obsDb.obsCount);
                for j = 1:obsDb.obsCount
                    obsTypes(j) = double(obsDb.obs(j).type);
                end
                [gyroDataNow, accelDataNow] = last_imu(measDb);
                b_omega = x(sMap.B_W);
                s_omega = x(sMap.S_W);
                b_f = x(sMap.B_F);
                s_f = x(sMap.S_F);
                omega_is = (s_omega .* gyroDataNow.omega_is) + b_omega;
                f_s = (s_f .* accelDataNow.f_b) + b_f;
                C_ce = lib_quatToDcm(x(sMap.Q_CS)) * (lib_quatToDcm(x(sMap.Q_ES)))';
                v_c = C_ce * x(sMap.V_E);
                H_nhc = zeros(3, esMap.LEN);
                H_nhc(1:3, esMap.V_E) = C_ce;
                H_nhc(1:3, esMap.PSI_EE) = C_ce * lib_skewMat(x(sMap.V_E));
                H_nhc(1:3, esMap.PSI_CC) = lib_skewMat(-v_c);
                meas_var_nhc = [(0.1)^2, (0.05)^2];
                nhcGate = norm(omega_is) < conf.NHC_RATE_THRES && abs(norm(f_s) - 9.81) < conf.NHC_ACCEL_THRES;
                rejectVelLat = false;
                rejectVelVer = false;
                if nhcGate
                    rejectVelLat = lib_testChi2(-v_c(2), P, H_nhc(2, :), meas_var_nhc(1));
                    rejectVelVer = lib_testChi2(-v_c(3), P, H_nhc(3, :), meas_var_nhc(2));
                end
                out = struct();
                out.time_s = t;
                out.ttag_us = double(filTtag);
                out.obs_types = obsTypes;
                out.pos_ecef_pre = x(sMap.POS_E)';
                out.vel_ecef_pre = x(sMap.V_E)';
                out.q_es_pre = x(sMap.Q_ES)';
                out.q_cs_pre = x(sMap.Q_CS)';
                out.omega_is_pre = omega_is';
                out.omega_norm_pre = norm(omega_is);
                out.f_s_pre = f_s';
                out.f_s_norm_pre = norm(f_s);
                out.v_c_pre = v_c';
                out.nhc_gate_pre = nhcGate;
                out.nhc_reject_y_pre = rejectVelLat;
                out.nhc_reject_z_pre = rejectVelVer;
                out.gps_residual_pre = gpsResidual;
                out.gps_variance_pre = gpsVariance;
                out.gps_h_rows_pre = H(1:min(3, size(H, 1)), :);
                out.p_pos_psiee_pre = P(esMap.POS_E, esMap.PSI_EE);
                out.p_vel_psiee_pre = P(esMap.V_E, esMap.PSI_EE);
                out.p_psiee_pre = P(esMap.PSI_EE, esMap.PSI_EE);
                [dx, Ppost] = lib_kfUpdateJoseph(P, obsDb, H);
                out.dx_psi_cc = dx(esMap.PSI_CC)';
                out.dx_psi_ee = dx(esMap.PSI_EE)';
                xPost = x;
                xPost(sMap.POS_E) = xPost(sMap.POS_E) + dx(esMap.POS_E);
                xPost(sMap.V_E) = xPost(sMap.V_E) + dx(esMap.V_E);
                xPost(sMap.B_F) = xPost(sMap.B_F) + dx(esMap.B_F);
                xPost(sMap.S_F) = xPost(sMap.S_F) + dx(esMap.S_F);
                xPost(sMap.B_W) = xPost(sMap.B_W) + dx(esMap.B_W);
                xPost(sMap.S_W) = xPost(sMap.S_W) + dx(esMap.S_W);
                psi_ee = dx(esMap.PSI_EE);
                qTmp = lib_eulerToQuat(psi_ee(1), psi_ee(2), psi_ee(3));
                qTmp = lib_quatMult(qTmp, xPost(sMap.Q_ES));
                xPost(sMap.Q_ES) = qTmp / sqrt(qTmp' * qTmp);
                psi_cc = dx(esMap.PSI_CC);
                qTmp = lib_eulerToQuat(psi_cc(1), psi_cc(2), psi_cc(3));
                qTmp = lib_quatMult(qTmp, xPost(sMap.Q_CS));
                xPost(sMap.Q_CS) = qTmp / sqrt(qTmp' * qTmp);
                out.q_es_post = xPost(sMap.Q_ES)';
                out.q_cs_post = xPost(sMap.Q_CS)';
                out.pos_ecef_post = xPost(sMap.POS_E)';
                out.p_psi_cc_pre = P(esMap.PSI_CC, esMap.PSI_CC);
                out.p_psi_cc_post = Ppost(esMap.PSI_CC, esMap.PSI_CC);
                out.p_pos_psiee_post = Ppost(esMap.POS_E, esMap.PSI_EE);
                out.p_vel_psiee_post = Ppost(esMap.V_E, esMap.PSI_EE);
                out.p_psiee_post = Ppost(esMap.PSI_EE, esMap.PSI_EE);
                fprintf('%s\n', jsonencode(out));
                return;
            end

            [dx, P] = lib_kfUpdateJoseph(P, obsDb, H);
            [x, lastUsedGnssTtag] = correct_state(x, dx, lastUsedGnssTtag, filTtag, measDb, conf, sMap, esMap);
        case 2
            measDb = measDb.addData(accelData(nextAccel), SensorType.ACCEL);
            nextAccel = nextAccel + 1;
        case 3
            measDb = measDb.addData(gnssData(nextGnss), SensorType.GNSS);
            nextGnss = nextGnss + 1;
    end
end

if collectRows
    fid = fopen(outPath, 'w');
    fprintf(fid, '%s', jsonencode(rows));
    fclose(fid);
    return;
end

error('target step not found');
end

function [gyroData, accelData] = last_imu(measDb)
gyroData = measDb.gyroBuf.getLastData();
accelData = measDb.accelBuf.getLastData();
end

function [obsDb, H, gpsResidual, gpsVariance] = prep_obs(x, P, filTtag, lastUsedGnssTtag, measDb, conf, sMap, esMap)
obsDb = ObsDb(20);
H = zeros(20, esMap.LEN);
gpsResidual = [0, 0, 0];
gpsVariance = [0, 0, 0];

[obsDb, H, ~, gpsResidual, gpsVariance] = prep_gnss_update(obsDb, H, x, P, filTtag, lastUsedGnssTtag, measDb, sMap, esMap);
[obsDb, H] = prep_nhc_update(obsDb, H, x, P, measDb, conf, sMap, esMap);
end

function [obsDb, H, lastUsedGnssTtagNew, gpsResidual, gpsVariance] = prep_gnss_update(obsDb, H, x, P, filTtag, lastUsedGnssTtag, measDb, sMap, esMap)
gpsResidual = [0, 0, 0];
gpsVariance = [0, 0, 0];
lastUsedGnssTtagNew = lastUsedGnssTtag;

[indGnssData, status] = measDb.gnssBuf.getLastIndLe(filTtag);
if ~status
    return;
end
gnssData = measDb.gnssBuf.getData(indGnssData);
dTtag = lib_ttagDiffUint64(filTtag, gnssData.ttag);
if ~(dTtag >= 0 && dTtag < SfConst.ImuTimeout / 2 && gnssData.ttag ~= lastUsedGnssTtag)
    return;
end

x_e_meas = lib_llhToEcef(gnssData.lat, gnssData.lon, gnssData.height, Wgs84);
R_n = diag([gnssData.hAcc^2; gnssData.hAcc^2; (2.5 * gnssData.hAcc)^2]);
[lat, lon, ~] = lib_ecefToLlh(x(sMap.POS_E), Wgs84);
C_en = lib_dcmEcefToNed(lat, lon);
R_e = C_en * R_n * C_en';
U = chol(R_e);
T = (U')^-1;
x_e_meas = T * x_e_meas;
x_e_est = T * x(sMap.POS_E);
dTtag = 1e-6 * lib_ttagDiffUint64(filTtag, lastUsedGnssTtag);
if dTtag == 0 || dTtag >= 1
    dTtag = 1;
end
lastUsedGnssTtagNew = gnssData.ttag;

H_tmp = zeros(3, esMap.LEN);
H_tmp(1:3, esMap.POS_E) = eye(3);
H_tmp = T * H_tmp;
res = x_e_meas - x_e_est;
needToReject = lib_testChi2(res, P, H_tmp, eye(3));
gpsResidual = res';
gpsVariance = [1 / dTtag, 1 / dTtag, 1 / dTtag];
if needToReject
    return;
end
obsTypes = [ObsType.POS_E_X, ObsType.POS_E_Y, ObsType.POS_E_Z];
for i = 1:3
    [obsDb, ok] = obsDb.add(obsTypes(i), x_e_meas(i), x_e_est(i), 1 / dTtag);
    if ok
        H(obsDb.obsCount, :) = H_tmp(i, :);
    end
end
end

function [obsDb, H] = prep_nhc_update(obsDb, H, x, P, measDb, conf, sMap, esMap)
gyroData = measDb.gyroBuf.getLastData();
accelData = measDb.accelBuf.getLastData();
b_omega = x(sMap.B_W);
s_omega = x(sMap.S_W);
b_f = x(sMap.B_F);
s_f = x(sMap.S_F);
omega_is = (s_omega .* gyroData.omega_is) + b_omega;
f_s = (s_f .* accelData.f_b) + b_f;
if ~(norm(omega_is) < conf.NHC_RATE_THRES && abs(norm(f_s) - 9.81) < conf.NHC_ACCEL_THRES)
    return;
end
dTtag = 1 / double(conf.GYRO_FREQ);
if dTtag >= 1
    dTtag = 1;
end
C_ce = lib_quatToDcm(x(sMap.Q_CS)) * (lib_quatToDcm(x(sMap.Q_ES)))';
v_c = C_ce * x(sMap.V_E);
H_tmp = zeros(3, esMap.LEN);
H_tmp(1:3, esMap.V_E) = C_ce;
H_tmp(1:3, esMap.PSI_EE) = C_ce * lib_skewMat(x(sMap.V_E));
H_tmp(1:3, esMap.PSI_CC) = lib_skewMat(-v_c);
meas_var = [(0.1)^2, (0.05)^2];
needToRejectVelLat = lib_testChi2(-v_c(2), P, H_tmp(2, :), meas_var(1));
needToRejectVelVer = lib_testChi2(-v_c(3), P, H_tmp(3, :), meas_var(2));
if ~needToRejectVelLat
    [obsDb, ok] = obsDb.add(ObsType.VEL_C_Y, 0, v_c(2), meas_var(1) / dTtag);
    if ok
        H(obsDb.obsCount, :) = H_tmp(2, :);
    end
end
if ~needToRejectVelVer
    [obsDb, ok] = obsDb.add(ObsType.VEL_C_Z, 0, v_c(3), meas_var(2) / dTtag);
    if ok
        H(obsDb.obsCount, :) = H_tmp(3, :);
    end
end
end

function [x, lastUsedGnssTtag] = correct_state(x, dx, lastUsedGnssTtag, filTtag, measDb, conf, sMap, esMap)
% Update the GNSS-used timestamp only if a fresh GNSS sample was eligible this epoch.
[indGnssData, status] = measDb.gnssBuf.getLastIndLe(filTtag);
if status
    gnssData = measDb.gnssBuf.getData(indGnssData);
    dTtag = lib_ttagDiffUint64(filTtag, gnssData.ttag);
    if dTtag >= 0 && dTtag < SfConst.ImuTimeout / 2 && gnssData.ttag ~= lastUsedGnssTtag
        lastUsedGnssTtag = gnssData.ttag;
    end
end

x(sMap.POS_E) = x(sMap.POS_E) + dx(esMap.POS_E);
x(sMap.V_E) = x(sMap.V_E) + dx(esMap.V_E);
x(sMap.B_F) = x(sMap.B_F) + dx(esMap.B_F);
x(sMap.S_F) = x(sMap.S_F) + dx(esMap.S_F);
x(sMap.B_W) = x(sMap.B_W) + dx(esMap.B_W);
x(sMap.S_W) = x(sMap.S_W) + dx(esMap.S_W);
psi_ee = dx(esMap.PSI_EE);
qTmp = lib_eulerToQuat(psi_ee(1), psi_ee(2), psi_ee(3));
qTmp = lib_quatMult(qTmp, x(sMap.Q_ES));
x(sMap.Q_ES) = qTmp / sqrt(qTmp' * qTmp);
psi_cc = dx(esMap.PSI_CC);
qTmp = lib_eulerToQuat(psi_cc(1), psi_cc(2), psi_cc(3));
qTmp = lib_quatMult(qTmp, x(sMap.Q_CS));
x(sMap.Q_CS) = qTmp / sqrt(qTmp' * qTmp);
end

function out = canonical_obs_types(rawObsTypes)
out = rawObsTypes;
for i = 1:length(out)
    switch out(i)
        case 7
            out(i) = 1;
        case 8
            out(i) = 2;
        case 9
            out(i) = 3;
        case 11
            out(i) = 4;
        case 12
            out(i) = 5;
    end
end
end

function gyroData = parse_gyro_data(inputDir)
file = dir(fullfile(inputDir, '*_Gyro.csv'));
raw = readmatrix(fullfile(file(1).folder, file(1).name), 'Delimiter', ';', 'NumHeaderLines', 3);
gyroData = repmat(GyroData, 1, size(raw, 1));
for i = 1:size(raw, 1)
    gyroData(i).valid = true;
    gyroData(i).ttag = uint64(floor(raw(i, 1) / 1000.0));
    gyroData(i).omega_is = raw(i, 2:4)';
end
end

function accelData = parse_accel_data(inputDir)
file = dir(fullfile(inputDir, '*_Acc.csv'));
raw = readmatrix(fullfile(file(1).folder, file(1).name), 'Delimiter', ';', 'NumHeaderLines', 3);
accelData = repmat(AccelData, 1, size(raw, 1));
for i = 1:size(raw, 1)
    accelData(i).valid = true;
    accelData(i).ttag = uint64(floor(raw(i, 1) / 1000.0));
    accelData(i).f_b = raw(i, 2:4)';
end
end

function gnssData = parse_gnss_data(inputDir)
file = dir(fullfile(inputDir, '*_GNSS.csv'));
raw = readmatrix(fullfile(file(1).folder, file(1).name), 'Delimiter', ';', 'NumHeaderLines', 1);
gnssData = repmat(GnssNavData, 1, size(raw, 1));
for i = 1:size(raw, 1)
    gnssData(i).valid = true;
    gnssData(i).ttag = uint64(floor(raw(i, 1) / 1000.0));
    gnssData(i).utcTime = uint64(raw(i, 2));
    gnssData(i).lat = pi / 180 * raw(i, 3);
    gnssData(i).lon = pi / 180 * raw(i, 4);
    gnssData(i).height = raw(i, 5);
    gnssData(i).hAcc = raw(i, 8);
    gnssData(i).vAcc = raw(i, 9);
end
end

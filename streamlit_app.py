import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings
from scipy.signal import find_peaks,resample


# Set up warnings and pandas options
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Initialize Streamlit app
st.set_page_config(page_title='Gait Processing UVA - Multifile', layout='wide')

##################################################### Functions #######################################################################################
def create_variables(dataframe):
    '''
    Takes in dataframe from The MotionMonitor Report. 

    Removes extra columns like Sample # and Unnamed : ##.
    Drops rows with missing data.
    Makes vGRF positive.
    Generates stance variables based on threshold.
    Generates heel strike and toe off variables based on a consistent flip between stance (more than 2 samples on either side).

    Returns the cleaned dataset.
    '''
    
    def remove_unnamed_columns(df):
        cleaned_df = df.loc[:, ~df.columns.str.contains(':')]
        cleaned_df = df.loc[:, ~df.columns.str.contains('Sample')]
        return cleaned_df
        
    dataframe = remove_unnamed_columns(dataframe)
    dataframe.drop(list(dataframe.filter(regex=':')), axis=1, inplace=True)
    
    
    # Remove rows where 'LeftPlateGRF' or 'RightPlateGRF' is NaN
    dataframe = dataframe.dropna(subset=['LeftPlateGRF', 'RightPlateGRF'])


    # vGRF is typically negative in The MotionMonitor, so we will make it positive for the purposes of graphing and interpretation.
    if dataframe['LeftPlateGRF'].mean() < 0:
        dataframe['LeftPlateGRF'] = dataframe['LeftPlateGRF']*-1
    if dataframe['RightPlateGRF'].mean() < 0:
        dataframe['RightPlateGRF'] = dataframe['RightPlateGRF']*-1
    
    # Is GRF Calibrated? If not we need to subtract the minimum value from the entire column. 
    # We exclude the first and last 5 samples to avoid noise.
    if dataframe['LeftPlateGRF'][5:-5].min() > 0:
        dataframe['LeftPlateGRF'] = dataframe['LeftPlateGRF']-dataframe['LeftPlateGRF'][5:-5].min()
    if dataframe['RightPlateGRF'][5:-5].min() > 0:
        dataframe['RightPlateGRF'] = dataframe['RightPlateGRF']-dataframe['RightPlateGRF'][5:-5].min()

    # Add stance 
    dataframe['RightStance'] = np.where(dataframe['RightPlateGRF']>force_threshold,1,0)
    dataframe['LeftStance'] = np.where(dataframe['LeftPlateGRF']>force_threshold,1,0)


    # Identify events where forceplate goes from loaded to unloaded, or unloaded to loaded
    dataframe['LeftHeelStrike'] = (dataframe['LeftStance']== 1) & (dataframe['LeftStance'].shift(1)== 0) & (dataframe['LeftStance'].shift(-1)== 1) & (dataframe['LeftStance'].shift(2)== 0)
    dataframe['LeftToeOff'] = (dataframe['LeftStance']== 0) & (dataframe['LeftStance'].shift(1)== 1)  & (dataframe['LeftStance'].shift(2)==1) & (dataframe['LeftStance'].shift(3)==1) 

    dataframe['RightHeelStrike'] = (dataframe['RightStance']== 1) & (dataframe['RightStance'].shift(1)== 0) & (dataframe['RightStance'].shift(-1)== 1) & (dataframe['RightStance'].shift(2)== 0)
    dataframe['RightToeOff'] = (dataframe['RightStance']== 0) & (dataframe['RightStance'].shift(1)== 1)  & (dataframe['RightStance'].shift(2)==1) & (dataframe['RightStance'].shift(3)==1) 
    return dataframe


##### New Functions####
def ordered_stepfinder(dataframe):
    #st.write('---')
    #st.write('Filename:',filename)
    # Initialize data
    left_HS_idx = dataframe.index[dataframe['LeftHeelStrike']].tolist()
    left_TO_idx = dataframe.index[dataframe['LeftToeOff']].tolist()
    right_HS_idx = dataframe.index[dataframe['RightHeelStrike']].tolist()
    right_TO_idx = dataframe.index[dataframe['RightToeOff']].tolist()

    if not left_HS_idx or not right_HS_idx:
        st.write('Missing step indices on one side')
        return []  # Return an empty list if no steps detected on either side

    # Find initial side with the first step
    first_side = 'Left' if min(left_HS_idx, default=float('inf')) < min(right_HS_idx, default=float('inf')) else 'Right'

    ordered_steps = []
    while left_HS_idx or right_HS_idx:
        current_hs_idx = left_HS_idx.pop(0) if first_side == 'Left' and left_HS_idx else (right_HS_idx.pop(0) if right_HS_idx else None)
        if current_hs_idx is None:
            break

        earliest = current_hs_idx + (400 / (1000 / Sampling_Rate))
        latest = current_hs_idx + Sampling_Rate
        current_to_idx = next((to for to in (left_TO_idx if first_side == 'Left' else right_TO_idx) if earliest < to < latest), None)

        if current_to_idx:
            ordered_steps.append((dataframe[current_hs_idx:current_to_idx], first_side))
            # Remove the used toe off index
            if first_side == 'Left':
                left_TO_idx.remove(current_to_idx)
            else:
                right_TO_idx.remove(current_to_idx)
        else:
            #st.write(f"No valid toe off found for heel strike at {current_hs_idx}")
            pass
            
        # Determine the side for the next step
        if left_HS_idx and right_HS_idx:
            first_side = 'Left' if left_HS_idx[0] < right_HS_idx[0] else 'Right'
        elif left_HS_idx:
            first_side = 'Left'
        elif right_HS_idx:
            first_side = 'Right'
        else:
            break  # No more heel strikes on either side

    return ordered_steps
# def ordered_stepfinder(dataframe):
#     # Initialize data
#     left_step_data = []
#     left_HS_idx = dataframe.index[dataframe['LeftHeelStrike']].tolist()
#     left_TO_idx = dataframe.index[dataframe['LeftToeOff']].tolist()

#     right_step_data = []
#     right_HS_idx = dataframe.index[dataframe['RightHeelStrike']].tolist()
#     right_TO_idx = dataframe.index[dataframe['RightToeOff']].tolist()

#     # Find initial side with the first step
#     first_side = 'Left' if min(left_HS_idx) < min(right_HS_idx) else 'Right'

#     # Initialize variables to track the heel strike and toe off indices
#     if first_side == 'Left':
#         current_hs_idx = left_HS_idx[0]
#         current_to_idx = next((to for to in left_TO_idx if to > current_hs_idx), None)
#     else:
#         current_hs_idx = right_HS_idx[0]
#         current_to_idx = next((to for to in right_TO_idx if to > current_hs_idx), None)

#     ordered_steps = []

#     # Main loop to find and order steps
#     while True:
#         if first_side == 'Left':
#             ordered_steps.append((dataframe[current_hs_idx:current_to_idx], 'Left'))
#             # Find the next right step that starts during the current left step
#             right_hs_idx = next((hs for hs in right_HS_idx if current_hs_idx <= hs < current_to_idx), None)
#             if right_hs_idx is None:
#                 break  # Exit if no more steps found
#             current_to_idx = next((to for to in right_TO_idx if to > right_hs_idx), None)
#             first_side = 'Right'  # Switch side
#         else:
#             ordered_steps.append((dataframe[current_hs_idx:current_to_idx], 'Right'))
#             # Find the next left step that starts during the current right step
#             left_hs_idx = next((hs for hs in left_HS_idx if current_hs_idx <= hs < current_to_idx), None)
#             if left_hs_idx is None:
#                 break  # Exit if no more steps found
#             current_to_idx = next((to for to in left_TO_idx if to > left_hs_idx), None)
#             first_side = 'Left'  # Switch side

#         current_hs_idx = left_hs_idx if first_side == 'Left' else right_hs_idx

#     return ordered_steps

def remove_bad_steps_with_neighbours(ordered_steps):
    bad_step_indices = set()

    # Identify bad steps
    for i, (step_data, side) in enumerate(ordered_steps):
        try:
            step_data = step_data.reset_index(drop=True)
            if side == 'Left':
                peaks, _ = find_peaks(step_data['LeftPlateGRF'], distance=150/(1000/Sampling_Rate), height=0.7*weight_N)
                if len(peaks) < 2 or (step_data['RightPlateGRF'][peaks[0]:peaks[1]].mean()/weight_N >= 0.025):
                    bad_step_indices.add(i)
            else:
                peaks, _ = find_peaks(step_data['RightPlateGRF'], distance=150/(1000/Sampling_Rate), height=0.7*weight_N)
                if len(peaks) < 2 or (step_data['LeftPlateGRF'][peaks[0]:peaks[1]].mean()/weight_N >= 0.025):
                    bad_step_indices.add(i)
        except Exception as e:
            print(f"Error processing step: {e}")

    # Extend the set of bad step indices to include the step before and after
    extended_bad_indices = set()
    print('Bad Steps', bad_step_indices)
    for idx in bad_step_indices:
        extended_bad_indices.update({idx, idx + 1}) # Just removes after
        # extended_bad_indices.update({idx - 1, idx, idx + 1}) Removes before and after
    
    # Ensure the indices are within the bounds of the ordered_steps list
    extended_bad_indices = {idx for idx in extended_bad_indices if 0 <= idx <= len(ordered_steps)}
    print('Extended Bad Steps', extended_bad_indices)
    # Filter out bad steps including their neighbours
    good_steps = [step for i, step in enumerate(ordered_steps) if i not in extended_bad_indices]

    return good_steps
def spatio_calc2(df, column):
    basecols = ['contact_time','time_to_peak','time_between_peaks','loading_rate','loading_rate_norm','first_peak_grf','first_peak_grf_norm','second_peak_grf','second_peak_grf_norm','grf_impulse','grf_impulse_norm','mean_grf','mean_grf_norm']
    grf_stats = pd.DataFrame(columns=basecols)
    contact_time = len(df[column]) / Sampling_Rate
    peaks, _ = find_peaks(df[column], distance=150/(1000/Sampling_Rate), height=0.7*weight_N)

    if len(peaks) > 0:
        time_to_peak = peaks[0] / Sampling_Rate
        time_between_peaks = (peaks[-1] - peaks[0]) / Sampling_Rate if len(peaks) > 1 else np.nan
        loading_rate = df[column].iloc[peaks[0]] / time_to_peak if time_to_peak > 0 else np.nan
        loading_rate_norm = loading_rate / weight_N if time_to_peak > 0 else np.nan
        first_peak_grf = df[column].iloc[peaks[0]]
        first_peak_grf_norm = first_peak_grf / weight_N
        second_peak_grf = df[column].iloc[peaks[-1]]
        second_peak_grf_norm = second_peak_grf/ weight_N
        mean_grf = df[column].mean()
        mean_grf_norm = mean_grf/ weight_N
        grf_impulse = np.trapz(df[column][df[column]>0], dx= (1000/Sampling_Rate)/ 1000)
        grf_impulse_norm = grf_impulse / weight_N
    else:
        time_to_peak, time_between_peaks, loading_rate, loading_rate_norm, first_peak_grf, first_peak_grf_norm , second_peak_grf, second_peak_grf_norm, mean_grf, mean_grf_norm, grf_impulse, grf_impulse_norm= [np.nan] * 13

    grf_stats.loc[0] = [contact_time,time_to_peak,time_between_peaks,loading_rate,loading_rate_norm, first_peak_grf, first_peak_grf_norm, second_peak_grf, second_peak_grf_norm,grf_impulse,grf_impulse_norm,mean_grf,mean_grf_norm]
    return grf_stats
def kinetic_calc2(df, column):
    clean_column = column.replace('Left','').replace('Right','')
    basecols = ['pos_peak', 'neg_peak', 'pos_peak_norm', 'neg_peak_norm', 'mean', 'mean_norm', 'positive_impulse', 'negative_impulse', 'positive_impulse_norm', 'negative_impulse_norm']
    newcols = [clean_column + '_' + item for item in basecols]
    kinetic_stats = pd.DataFrame(columns = newcols)

    # Initialize variables to ensure they have values even if try block fails
    pos_peak = neg_peak = pos_peak_norm = neg_peak_norm = np.nan

    try:
        # Find the positive peak in the second half of the waveform
        l = len(df[column])
        first_half = df[column][:round(l/2)]
        second_half = df[column][round(l/2):]

        pos_peaks, _ = find_peaks(second_half)#, distance=len(second_half)-1
        if pos_peaks.size > 0:
            #pos_peak = df[column].iloc[pos_peaks+round(len(second_half))].max()
            pos_peak = second_half.iloc[pos_peaks].max()
            pos_peak_norm = pos_peak / weight_N

        # Find the negative peak in the first half of the waveform
        
        neg_peaks, _ = find_peaks(-1 * first_half)
        if neg_peaks.size > 0:
            neg_peak = first_half.iloc[neg_peaks].min()
            neg_peak_norm = neg_peak / weight_N
    except Exception as e:
        print(f"Error in kinetic calculation: {e}")

    mean = df[column].mean()
    mean_norm = mean / weight_N
    positive_impulse = np.trapz(df[column][df[column] > 0], dx=(1000/Sampling_Rate)/1000)
    negative_impulse = np.trapz(df[column][df[column] < 0], dx=(1000/Sampling_Rate)/1000)
    positive_impulse_norm = positive_impulse / weight_N
    negative_impulse_norm = negative_impulse / weight_N

    kinetic_stats.loc[0] = [pos_peak, neg_peak, pos_peak_norm, neg_peak_norm, mean, mean_norm, positive_impulse, negative_impulse, positive_impulse_norm, negative_impulse_norm]
    return kinetic_stats
def kinematic_calc2(df, column):
    """
    Calculate max and min for kinematic variables.

    Parameters:
    - df: pandas DataFrame containing the data.
    - column: string, the column name for kinematic variable.

    Returns:
    - DataFrame with max and min values for the specified kinematic variable.
    """
    clean_column = column.replace('Left','').replace('Right','')
    basecols = ['angle_max', 'angle_min', 'angle_at_ic']
    newcols = [clean_column + '_' + item for item in basecols]
    kinematic_stats = pd.DataFrame(columns = newcols)

    varmax = df[column].max()
    varmin = df[column].min()
    icval = df[column].iloc[0]  # Using .iloc to access by integer location
    kinematic_stats.loc[0] = [varmax, varmin, icval]
    return kinematic_stats

def new_step_summarizer(ordered_steps):
    all_steps = pd.DataFrame()

    for step_data, side in ordered_steps:
        if side == 'Left':
            kinetic_vars = left_kinetic_vars
            kinematic_vars = left_kinematic_vars
        else:
            kinetic_vars = right_kinetic_vars
            kinematic_vars = right_kinematic_vars

        # Calculate spatiotemporal data
        grf_column = f'{side}PlateGRF'
        spatio_step = spatio_calc2(step_data, grf_column)

        # Calculate kinetic and kinematic data
        kinetic_step = pd.DataFrame()
        for col in kinetic_vars:
            kin = kinetic_calc2(step_data, col)
            kinetic_step = pd.concat([kinetic_step, kin], axis=1)

        kinematic_step = pd.DataFrame()
        for col in kinematic_vars:
            kin = kinematic_calc2(step_data, col)
            kinematic_step = pd.concat([kinematic_step, kin], axis=1)

        # Combine all data for the current step
        step_summary = pd.concat([spatio_step, kinetic_step, kinematic_step], axis=1)
        step_summary['Limb'] = side

        # Append to the overall data
        all_steps = pd.concat([all_steps, step_summary])

    return all_steps
def perc_diff(df,column):
    left = df[df['Limb'] == 'Left'][column].mean()
    right = df[df['Limb'] == 'Right'][column].mean()
    abs_diff = abs(left-right)
    avg = (left+right)/2
    perc_diff = (abs_diff/avg)*100
    return perc_diff
############################     App    #########################################

# Streamlit UI for file upload
st.title('Gait Processing Code')
st.write('This app is allows the upload of multiple reports from The MotionMonitor. This will accept KitchenSink or CADReport Files, and will process data accordingly.')
st.write('')
data_files = st.file_uploader('Upload the text file reports from The Motion Monitor', accept_multiple_files=True)

# Initialize an empty DataFrame to store all summaries
combined_summaries = pd.DataFrame()
limb_summaries = pd.DataFrame()
limb_names = pd.DataFrame()
left_stretched_series_ankle = []
right_stretched_series_ankle = []
left_stretched_series_knee = []
right_stretched_series_knee = []
left_stretched_series_hip = []
right_stretched_series_hip = []
# Check if files have been uploaded
if data_files:
    force_threshold = st.sidebar.number_input('Force Threshold (N): ', min_value=20, value=50, max_value=100)
    normalize = st.sidebar.selectbox('What would you like to normalize by?',('Newtons','Kilograms'))
    # Process each file
    for data_file in data_files:
        # Read metadata from the file
        data = pd.read_csv(data_file, nrows = 5)

        
        # Extract metadata
        filename = data.iloc[1, 0].split('\t')[0]
        Sampling_Rate = float(data.iloc[2, 0].split('//')[0])
        Capture_Period = float(data.iloc[3, 0].split('//')[0])
        weight_kg = float(data.iloc[4, 0].split('//')[0])
        weight_N = 9.81 * weight_kg

        if normalize == 'Newtons':
            normalizer = weight_N
        else:
            normalizer = weight_kg

        # Display metadata in sidebar
        st.sidebar.write(f'Filename: {filename}')
        st.sidebar.write(f'Sampling Rate (Hz): {Sampling_Rate}')
        st.sidebar.write(f'Capture Period (sec): {Capture_Period}')
        st.sidebar.write(f'Subject Weight (kg): {weight_kg}')

        data_file.seek(0)
        # Read the actual data from the file, skipping metadata lines
        df = pd.read_csv(data_file, delimiter='\t', skiprows=9)

        # # Process the data
        df_processed = create_variables(df)

        # Get ordered steps
        os = ordered_stepfinder(df_processed)
        
        # Remove the bad steps
        gs = remove_bad_steps_with_neighbours(os)
        
        if len(gs)>0:


            if 'RightKneeRotation' in gs[0][0].columns:
                left_kinetic_vars = ['LeftKneeMomentZ','LeftKneeMomentY','LeftKneeMomentX', 'LeftKneeForceZ','LeftKneeForceY','LeftKneeForceX','LeftHipMomentZ','LeftHipMomentY','LeftHipMomentX',  'LeftHipForceZ','LeftHipForceY','LeftHipForceX', 'LeftAnkleMomentZ','LeftAnkleMomentY','LeftAnkleMomentX',   'LeftAnkleForceZ','LeftAnkleForceY','LeftAnkleForceX']
                right_kinetic_vars = ['RightKneeMomentZ','RightKneeMomentY','RightKneeMomentX', 'RightKneeForceZ','RightKneeForceY','RightKneeForceX','RightHipMomentZ','RightHipMomentY','RightHipMomentX','RightHipForceZ','RightHipForceY','RightHipForceX', 'RightAnkleMomentZ','RightAnkleMomentY','RightAnkleMomentX','RightAnkleForceZ','RightAnkleForceY','RightAnkleForceX']
                left_kinematic_vars = ['LeftKneeRotation','LeftKneeFlexion','LeftHipRotation','LeftHipFlexion','LeftHipAbduction','LeftAnkleInversion','LeftAnkleFlexion','LeftAnkleAbduction']
                right_kinematic_vars = ['RightKneeRotation','RightKneeFlexion','RightHipRotation','RightHipFlexion','RightHipAbduction','RightAnkleInversion','RightAnkleFlexion','RightAnkleAbduction']
            else:
                left_kinetic_vars = ['LeftKneeMomentX','LeftKneeMomentY','LeftKneeMomentZ', 'LeftHipMomentX','LeftHipMomentY','LeftHipMomentZ','LeftAnkleMomentX','LeftAnkleMomentY','LeftAnkleMomentZ']
                right_kinetic_vars = ['RightKneeMomentX','RightKneeMomentY','RightKneeMomentZ', 'RightHipMomentX','RightHipMomentY','RightHipMomentZ','RightAnkleMomentX','RightAnkleMomentY','RightAnkleMomentZ']
                left_kinematic_vars = ['LeftKneeFlexion','LeftHipFlexion','LeftHipAbduction','LeftAnkleFlexion']
                right_kinematic_vars = ['RightKneeFlexion','RightHipFlexion','RightHipAbduction','RightAnkleFlexion']
            # Generate variables
            all_steps = new_step_summarizer(gs)

            limb_steps = all_steps.groupby(['Limb']).agg('mean').reset_index()
            # April 30 Test
            # Need to add a display of limb aggregate data to examine asymmetry
        
            trial = 'C' + filename.split('_')[-1]  # Extract trial from the filename
            limb_copy = limb_steps.copy()
            limb_names['Trial'] = trial
            limb_names['Filename'] = '_'.join(filename.split('_')[:-1])  # Correct filename to be the subject identifier
            limb_copy['Trial'] = trial
            limb_copy['Filename'] = '_'.join(filename.split('_')[:-1])  # Correct filename to be the subject identifier
            limb_summaries = pd.concat([limb_names, limb_copy,limb_summaries], ignore_index=True)
            # April 30, make a plot for all ankle moments
            for(step_data, side) in gs:
                if step_data.isnull().values.any():
                    pass
                else:
                    if side == 'Left':
                        left_resampled_ankle = resample(step_data['LeftAnkleMomentX']/normalizer,100)    
                        left_stretched_series_ankle.append(left_resampled_ankle)

                        left_resampled_knee = resample(step_data['LeftKneeMomentX']/normalizer,100)
                        left_stretched_series_knee.append(left_resampled_knee)

                        left_resampled_hip = resample(step_data['LeftHipMomentX']/normalizer,100)
                        left_stretched_series_hip.append(left_resampled_hip)
                    else:
                        right_resampled_ankle = resample(step_data['RightAnkleMomentX']/normalizer,100)
                        right_stretched_series_ankle.append(right_resampled_ankle)

                        right_resampled_knee = resample(step_data['RightKneeMomentX']/normalizer,100)
                        right_stretched_series_knee.append(right_resampled_knee)

                        right_resampled_hip = resample(step_data['RightHipMomentX']/normalizer,100)
                        right_stretched_series_hip.append(right_resampled_hip)            
                
            # End of April 30 Test
            nolimb = all_steps.drop(columns='Limb')

            final_summary = nolimb.mean()
            
            # Correct the 'Filename' assignment
            trial = 'C' + filename.split('_')[-1]  # Extract trial from the filename
            final_summary['Trial'] = trial
            final_summary['Filename'] = '_'.join(filename.split('_')[:-1])  # Correct filename to be the subject identifier

            # Convert final_summary Series to DataFrame for a single row
            final_summary_df = pd.DataFrame([final_summary])
            
            # Append this file's summary to the combined DataFrame
            combined_summaries = pd.concat([combined_summaries, final_summary_df], ignore_index=True)
            combined_summaries = combined_summaries[['Trial'] + [col for col in combined_summaries.columns if col != 'Trial']]
            combined_summaries = combined_summaries[['Filename'] + [col for col in combined_summaries.columns if col != 'Filename']]
            combined_summaries.sort_values(by=['Filename','Trial'], inplace=True)


        else:
            st.write(f'No good steps found for {filename}. There were {len(os)} ordered steps.')
            st.line_chart(df_processed[['LeftPlateGRF','RightPlateGRF']], color = ["#232D4B","#E57200"])
            st.write('---')
            trial = 'C' + filename.split('_')[-1]
            subject = '_'.join(filename.split('_')[:-1])
            skipped_data = pd.DataFrame({'Filename': [subject],
                            'Trial': [trial]})
            combined_summaries = pd.concat([combined_summaries, skipped_data], axis=0, ignore_index=True)
            continue

    # Once all files are processed, display the combined DataFrame
    st.write('Combined Trial Summaries')
    st.dataframe(combined_summaries)
    
    # April 30 Test Outputs
    st.write('---')
    st.write('Limb Summaries')
    st.dataframe(limb_summaries.sort_values(by=['Filename','Trial','Limb']))

    st.write('---')
    # st.write('Sagittal Plane Knee Moment')
    # st.write('Positive')
    # plot1_total_pos = sns.barplot(data = limb_summaries, x = 'Limb', y = 'KneeMomentX_pos_peak_norm',dodge=False,legend=False)
    # st.pyplot(plot1_total_pos.get_figure(),clear_figure=True)
    # st.write('Percent Diff',perc_diff(limb_summaries,'KneeMomentX_pos_peak_norm'),'%')
    # plot1 = sns.barplot(data = limb_summaries, x = 'Limb', y = 'KneeMomentX_pos_peak_norm', hue = 'Trial')
    # st.pyplot(plot1.get_figure(),clear_figure=True)
    # st.write('Negative')
    # plot2_total_neg = sns.barplot(data = limb_summaries, x = 'Limb', y = 'KneeMomentX_neg_peak_norm',dodge=False,legend=False)
    # st.pyplot(plot2_total_neg.get_figure(),clear_figure=True)
    # st.write('Percent Diff',perc_diff(limb_summaries,'KneeMomentX_neg_peak_norm'),'%')
    # plot2 = sns.barplot(data = limb_summaries, x = 'Limb', y = 'KneeMomentX_neg_peak_norm', hue = 'Trial')
    # st.pyplot(plot2.get_figure(),clear_figure=True)

    # st.write('---')
    # st.write('Sagittal Plane Ankle Moment')
    # st.write('Positive')
        
    # plot3_total_pos = sns.barplot(data = limb_summaries, x = 'Limb', y = 'AnkleMomentX_pos_peak_norm',dodge=False,legend=False)
    # st.pyplot(plot3_total_pos.get_figure(),clear_figure=True)
    # st.write('Percent Diff',perc_diff(limb_summaries,'AnkleMomentX_pos_peak_norm'),'%')
    # plot3 = sns.barplot(data = limb_summaries, x = 'Limb', y = 'AnkleMomentX_pos_peak_norm', hue = 'Trial')
    # st.pyplot(plot3.get_figure(),clear_figure=True)
    # st.write('Negative')
    # plot3_total_neg = sns.barplot(data = limb_summaries, x = 'Limb', y = 'AnkleMomentX_neg_peak_norm',dodge=False,legend=False)
    # st.pyplot(plot3_total_neg.get_figure(),clear_figure=True)
    # st.write('Percent Diff',perc_diff(limb_summaries,'AnkleMomentX_neg_peak_norm'),'%')
    # plot4 = sns.barplot(data = limb_summaries, x = 'Limb', y = 'AnkleMomentX_neg_peak_norm', hue = 'Trial')
    # st.pyplot(plot4.get_figure(),clear_figure=True)

    st.write('---')
    left_average_series_ankle = np.mean(left_stretched_series_ankle, axis=0)
    right_average_series_ankle = np.mean(right_stretched_series_ankle, axis=0)  
    left_average_series_knee = np.mean(left_stretched_series_knee, axis=0)
    right_average_series_knee = np.mean(right_stretched_series_knee, axis=0)

    left_average_series_hip = np.mean(left_stretched_series_hip, axis=0)
    right_average_series_hip = np.mean(right_stretched_series_hip, axis=0)

    fig3,ax3 = plt.subplots()
    plt.title('Normalized Sagittal Plane Ankle Moment')
    x = np.linspace(0,1,100)
    ax3.plot(x,left_average_series_ankle, label='Left Ankle',color='red',linewidth=2)
    ax3.plot(x, right_average_series_ankle,label='Right Ankle',color='blue',linewidth=2)
    ax3.legend()

    left_pos_peaks, _ = find_peaks(left_average_series_ankle, distance=99)
    right_pos_peaks, _ = find_peaks(right_average_series_ankle, distance=99)

    left_neg_peaks, _ = find_peaks(-left_average_series_ankle, distance=99)
    right_neg_peaks, _ = find_peaks(-right_average_series_ankle, distance=99)

    ax3.plot(x[left_pos_peaks], left_average_series_ankle[left_pos_peaks], "x",color='red')
    ax3.plot(x[right_pos_peaks], right_average_series_ankle[right_pos_peaks], "x",color='blue')
    ax3.plot(x[left_neg_peaks], left_average_series_ankle[left_neg_peaks], "x",color='red')
    ax3.plot(x[right_neg_peaks], right_average_series_ankle[right_neg_peaks], "x",color='blue')
    ax3.axhline(y=0,color='black')
    st.pyplot(fig3,clear_figure=True)

    st.write('---')
    fig2,ax2 = plt.subplots()
    plt.title('Normalized Sagittal Plane Knee Moment')
    x = np.linspace(0,1,100)
    ax2.plot(x,left_average_series_knee, label='Left Knee',color='red',linewidth=2)
    ax2.plot(x,right_average_series_knee, label='Right Knee',color='blue',linewidth=2)
    ax2.legend()

    left_pos_peaks_back, _ = find_peaks(left_average_series_knee[50:100], distance=49)
    right_pos_peaks_back, _ = find_peaks(right_average_series_knee[50:100], distance=49)

    left_pos_peaks, _ = find_peaks(left_average_series_knee, distance=99)
    right_pos_peaks, _ = find_peaks(right_average_series_knee,distance=99)

    left_neg_peaks, _ = find_peaks(-left_average_series_knee, distance=99)
    right_neg_peaks, _ = find_peaks(-right_average_series_knee, distance=99)
    
    ax2.plot(x[left_pos_peaks_back+50], left_average_series_knee[left_pos_peaks_back+50], "x",color='red')
    ax2.plot(x[right_pos_peaks_back+50], right_average_series_knee[right_pos_peaks_back+50], "x",color='blue')

    ax2.plot(x[left_pos_peaks], left_average_series_knee[left_pos_peaks], "x",color='red')
    ax2.plot(x[right_pos_peaks], right_average_series_knee[right_pos_peaks], "x",color='blue')
    ax2.plot(x[left_neg_peaks], left_average_series_knee[left_neg_peaks], "x",color='red')
    ax2.plot(x[right_neg_peaks], right_average_series_knee[right_neg_peaks], "x",color='blue')

    ax2.axhline(y=0,color='black')
    st.pyplot(fig2, clear_figure=True)

    # st.write('---')
    # fig1,ax1 = plt.subplots()
    # #plt.title('Normalized Sagittal Plane Hip Moment')
    # x = np.linspace(0,1,100)
    # ax1.plot(x,left_average_series_hip, label='Left Hip',color='red',linewidth=1)
    # ax1.plot(x, right_average_series_hip,label='Right Hip',color='blue',linewidth=1)
    # ax1.legend()

    # left_pos_peaks, _ = find_peaks(left_average_series_hip, distance=99)
    # right_pos_peaks, _ = find_peaks(right_average_series_hip,distance=99)

    # left_neg_peaks, _ = find_peaks(-left_average_series_hip, distance=99)
    # right_neg_peaks, _ = find_peaks(-right_average_series_hip, distance=99)

    # ax1.plot(x[left_pos_peaks], left_average_series_hip[left_pos_peaks], "x",color='red')
    # ax1.plot(x[right_pos_peaks], right_average_series_hip[right_pos_peaks], "x",color='blue')
    # ax1.plot(x[left_neg_peaks], left_average_series_hip[left_neg_peaks], "x",color='red')
    # ax1.plot(x[right_neg_peaks], right_average_series_hip[right_neg_peaks], "x",color='blue')
    
    # st.pyplot(fig1,clear_figure=True)
else:
    st.error('Please upload the files corresponding to the trials.')




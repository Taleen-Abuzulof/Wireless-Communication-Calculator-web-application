import math
import os
import time
from flask import Flask, render_template, request, flash, redirect, url_for
from flask_wtf.csrf import CSRFProtect
import pandas as pd
from requests import session
from Gemini import GeminiAPIClient, AnalysisType
import Gemini
from forms import CellularSystem, CommunicationSystemForm, LinkBudgetForm, OfdmSystems
from decimal import Decimal , getcontext, InvalidOperation
import markdown
from llm_agent import GroqAPIClient, AnalysisType as AnalysisType2


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
csrf = CSRFProtect(app)
csrf = CSRFProtect(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/communication_system', methods=['GET', 'POST'])
def communication_system():
    form = CommunicationSystemForm()
    results = None
    explanation_groq = None
    explanation_gemini = None

    if form.validate_on_submit():
        getcontext().prec = 10

        try:
            # Convert and validate inputs
            bandwidth = Decimal(form.bandwidth.data)
            quantizer_bits = int(form.quantizer_bits.data)
            encoder_compression = Decimal(form.compression_rate.data)
            channel_encoder_rate = Decimal(form.channel_encoder_rate.data)
            voice_segment= Decimal(form.voice_segment.data)

            # Calculations
            sampling_freq = bandwidth * 2 * 1000
            quantizer_levels = 2 ** quantizer_bits
            quantizer_rate = (sampling_freq * quantizer_bits) / Decimal(1000)    #in kbps
            source_encoder_rate = quantizer_rate * encoder_compression    # kbps
            ##### test channel_encoder_rate_bits = Decimal(22.8)

            channel_encoder_rate_bits = source_encoder_rate / channel_encoder_rate #channel coding result 
            data_after_channel_coding = channel_encoder_rate_bits * voice_segment # to get the bits of data after channel coding kbps * ms 
 
            interleaver_rate_bits = channel_encoder_rate_bits
            
            burst_needed = (data_after_channel_coding / Decimal(57)) / Decimal(2)   # or we can put it directly / 114
            over_head_data = burst_needed * Decimal(42.2496) # 3 + 1 + 26 + 1 + 3 + 8.2496 
            total_bits_per_vseg = data_after_channel_coding + over_head_data 
            
            burst_formatting_rate = total_bits_per_vseg / voice_segment

            # Prepare input data and results
            data = {
                "bandwidth_kHz": float(bandwidth),
                "quantizer_bits": quantizer_bits,
                "compression_rate": float(encoder_compression),
                "channel_encoder_rate": float(channel_encoder_rate),
                "voice segment": float(voice_segment)
            }

            # Prepare results
            results = {
                'sampling_freq': round(sampling_freq, 3),
                'quantizer_levels': quantizer_levels,
                'quantizer_rate': round(quantizer_rate, 3),
                'source_encoder_rate': round(source_encoder_rate, 3),
                'channel_encoder_rate_bits': round(channel_encoder_rate_bits, 3),
                'interleaver_rate_bits': round(interleaver_rate_bits,3 ),
                'burst_formatting_rate': round(burst_formatting_rate, 3),
            }

            try:
                groq_client = GroqAPIClient()
                prompt = groq_client.forge_prompt({}, results, AnalysisType2.COMM_SYSTEM)
                explanation_groq = markdown.markdown(groq_client.generate_response(prompt))
            except Exception as e:
                print(f"Groq Error: {e}")
                explanation_groq = "AI analysis temporarily unavailable."

            # Gemini AI Explanation
            try:
                gemini_client = GeminiAPIClient()
                prompt = gemini_client.forge_prompt({}, results, AnalysisType.COMM_SYSTEM)
                explanation_gemini = markdown.markdown(gemini_client.generate_response(prompt))
            except Exception as e:
                print(f"Gemini Error: {e}")
                explanation_gemini = "AI analysis temporarily unavailable."

        except (InvalidOperation, ValueError, TypeError):
            flash("Please enter valid numeric values in all fields.")
            return redirect(url_for('communication_system'))


        return render_template('communication_system.html', form=form, results=results,
                            explanation_groq=explanation_groq,
                            explanation_gemini=explanation_gemini)


    return render_template('communication_system.html', form=form, results=results, explanation=None)


# Update your OFDM route to include rate limiting
@app.route('/ofdm_systems', methods=['GET', 'POST'])
def ofdm_systems():
    form = OfdmSystems()
    results = None
    
    if request.method == 'POST':
        if form.validate_on_submit():
            # Simple rate limiting check
            user_key = request.remote_addr
            now = time.time()
            
            if not hasattr(app, 'ofdm_submissions'):
                app.ofdm_submissions = {}
            
            if user_key in app.ofdm_submissions:
                if now - app.ofdm_submissions[user_key] < 8:  # 8 seconds cooldown
                    wait_time = int(8 - (now - app.ofdm_submissions[user_key]))
                    flash(f'Please wait {wait_time} seconds before submitting again.', 'warning')
                    return redirect(request.url)
            
            app.ofdm_submissions[user_key] = now
            
            try:
                form_data = {
                    'bandwidth': form.bandwidth.data,
                    'subcarrier_spacing': form.subcarrier_spacing.data,
                    'ofdm_symbols': form.ofdm_symbols.data,
                    'modulation_type': form.modulation_type.data,
                    'block_duration': form.block_duration.data,
                    'resource_blocks': form.resource_blocks.data,
                }
                
                results = calculate_ofdm_parameters(form_data)
                
                if results and 'Error' in results:
                    flash(f'Calculation error: {results["Error"]}', 'error')
                    results = None
                elif results and len(results) > 0:
                    flash('OFDM calculations completed successfully!', 'success')
                else:
                    flash('Calculation returned empty results', 'warning')
                    results = None
                
            except Exception as e:
                flash(f'Error in OFDM calculations: {str(e)}', 'error')
                results = None
        else:
            flash('Please correct the errors in the form below.', 'error')
    
    return render_template('ofdm_systems.html', form=form, results=results)

@app.route('/link_budget', methods=['GET', 'POST'])
def link_budget():
    form = LinkBudgetForm()
    results = {}
    explanation_gemini = None
    explanation_groq = None

    if form.validate_on_submit():
        def convertTodB(value, unit):
            if unit == 'Watt':
                return 10 * math.log10(value)
            elif unit == 'dBm':
                return value - 30
            return value  # assume dB

        path_loss = convertTodB(float(form.path_loss.data), form.path_loss_unit.data)
        frequency = float(form.frequency.data)
        transmitter_antenna_gain = convertTodB(float(form.transmitter_antenna_gain.data), form.transmitter_antenna_gain_unit.data)
        receiver_antenna_gain = convertTodB(float(form.receiver_antenna_gain.data), form.receiver_antenna_gain_unit.data)
        data_rate = convertTodB(float(form.data_rate.data) * 1000, "Watt")
        antenna_feed_line_loss = convertTodB(float(form.antenna_feed_line_loss.data), form.antenna_feed_line_loss_unit.data)
        other_losses = convertTodB(float(form.other_losses.data), form.other_losses_unit.data)
        fade_margin = convertTodB(float(form.fade_margin.data), form.fade_margin_unit.data)
        receiver_amplifier_gain = convertTodB(float(form.receiver_amplifier_gain.data), form.receiver_amplifier_gain_unit.data)
        transmitter_amplifier_gain = convertTodB(float(form.transmitter_amplifier_gain.data), form.transmitter_amplifier_gain_unit.data)
        noise_figure_total = convertTodB(float(form.noise_figure_total.data), form.noise_figure_total_unit.data)
        noise_temperature = float(form.noise_temperature.data)# Keep as Kelvin
        link_margin = convertTodB(float(form.link_margin.data), form.link_margin_unit.data)
        modulation = form.modulation.data
        max_ber = form.max_bit_error_rate.data

        eb_n0 = EB_N0_VALUES[modulation][max_ber]
        k_db = -228.6  # Boltzmann constant in dB
        T_dB  = 10 * math.log10(noise_temperature)

        power_received = k_db + T_dB + noise_figure_total + data_rate + eb_n0 + link_margin
        power_transmitted = power_received + path_loss + antenna_feed_line_loss + other_losses + fade_margin - transmitter_antenna_gain - transmitter_amplifier_gain - receiver_antenna_gain - receiver_amplifier_gain

        # power_transmitted = k_db + noise_temperature + noise_figure_total + data_rate + eb_n0 + link_margin
        # power_received = power_transmitted + path_loss + antenna_feed_line_loss + other_losses + fade_margin - transmitter_antenna_gain - transmitter_amplifier_gain - receiver_antenna_gain - receiver_amplifier_gain


        results = {
            'pr': round(power_received, 2),
            'transmit_power': round(power_transmitted, 2)
        }
    # Groq
        try:
            groq_client = GroqAPIClient()
            prompt = groq_client.forge_prompt({}, results, AnalysisType2.LINK_BUDGET)
            explanation_groq = markdown.markdown(groq_client.generate_response(prompt))
        except Exception as e:
            print(f"Groq Error: {e}")
            explanation_groq = "AI analysis temporarily unavailable."

        # Gemini
        try:
            gemini_client = GeminiAPIClient()
            prompt = gemini_client.forge_prompt({}, results, AnalysisType.LINK_BUDGET)
            explanation_gemini = markdown.markdown(gemini_client.generate_response(prompt))
        except Exception as e:
            print(f"Gemini Error: {e}")
            explanation_gemini = "AI analysis temporarily unavailable."

        return render_template(
            'link_budget.html',
            form=form,
            results=results,
            explanation_groq=explanation_groq,
            explanation_gemini=explanation_gemini
        )

    return render_template('link_budget.html', form=form, results=results, explanation=None)



@app.route('/cellular-system', methods=['GET', 'POST'])
def cellular_system():
    form = CellularSystem()
    results = None
    
    if form.validate_on_submit():
        try:
            # Extract form data
            form_data = {
                'times_slots_per_carrier': form.times_slots_per_carrier.data,
                'area': form.area.data,
                'number_of_users': form.number_of_users.data,
                'average_calls_per_day': form.average_calls_per_day.data,
                'average_call_duration': form.average_call_duration.data,
                'call_drop_probability': form.call_drop_probability.data,
                'traffic_model': form.traffic_model.data,  # New field
                'minimum_sir_required': form.minimum_sir_required.data,
                'measured_power_at_reference_distance': form.measured_power_at_reference_distance.data,
                'reference_distance': form.reference_distance.data,
                'path_loss_exponent': form.path_loss_exponent.data,
                'receiver_sensitivity': form.receiver_sensitivity.data,
            }
            
            # Perform calculations
            results = calculate_cellular_parameters(form_data)
            
            flash('Calculations completed successfully!', 'success')
            
        except Exception as e:
            flash(f'Error in calculations: {str(e)}', 'error')
    
    elif request.method == 'POST':
        flash('Please correct the errors in the form below.', 'error')
    
    return render_template('cellular_system.html', form=form, results=results)

# Load Erlang B table at startup
def load_erlang_b_table():
    """Load the Erlang B table from CSV file"""
    try:
        csv_path = 'Erlang B Table.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df
        else:
            print(f"Warning: {csv_path} not found. Channel calculations will use approximation.")
            return None
    except Exception as e:
        print(f"Error loading Erlang B table: {e}")
        return None

# Global variable to store the Erlang B table
ERLANG_B_TABLE = load_erlang_b_table()

def find_channels_from_erlang_b(traffic_erlangs, blocking_probability_percent):
    """Find minimum number of channels needed from Erlang B table"""
    if ERLANG_B_TABLE is None:
        return None
    
    try:
        available_probs = [0.1, 0.2, 0.5, 1, 1.2, 1.3, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50]
        closest_prob = min(available_probs, key=lambda x: abs(x - blocking_probability_percent))
        prob_column = str(closest_prob) if closest_prob != 5 else '5'
        
        if prob_column not in ERLANG_B_TABLE.columns:
            print(f"Warning: Column {prob_column} not found in Erlang B table")
            return None
        
        for index, row in ERLANG_B_TABLE.iterrows():
            try:
                capacity = float(row[prob_column])
                if capacity >= traffic_erlangs:
                    return int(row['N'])
            except (ValueError, TypeError):
                continue
        
        return int(ERLANG_B_TABLE['N'].iloc[-1]) + 10
        
    except Exception as e:
        print(f"Error in Erlang B lookup: {e}")
        return None

def erlang_c_channels(traffic_erlangs, target_probability):
    """
    Calculate minimum channels needed using Erlang C formula (iterative approach)
    For Erlang C (queuing model)
    """
    try:
        # Start with traffic + some margin as initial guess
        n = max(int(traffic_erlangs) + 1, 1)
        max_iterations = 100
        
        for iteration in range(max_iterations):
            if n <= traffic_erlangs:
                n += 1
                continue
            
            # Calculate Erlang C probability
            # P(delay) = [A^N / N!] * [N / (N - A)] / sum(k=0 to N-1)[A^k / k!] + [A^N / N!] * [N / (N - A)]
            
            # Calculate denominator sum
            sum_term = 0
            for k in range(n):
                sum_term += (traffic_erlangs ** k) / math.factorial(k)
            
            # Calculate numerator
            numerator = (traffic_erlangs ** n) / math.factorial(n) * (n / (n - traffic_erlangs))
            
            # Calculate Erlang C probability
            prob_delay = numerator / (sum_term + numerator)
            
            if prob_delay <= target_probability:
                return n
            
            n += 1
        
        # If no solution found within max_iterations, return approximation
        return int(traffic_erlangs * 1.5) + 5
        
    except Exception as e:
        print(f"Error in Erlang C calculation: {e}")
        return None

def convertToLinear(value):
    """Convert dB to linear scale"""
    return 10 ** (value / 10)

def calculate_cellular_parameters(form_data):
    """
    Perform cellular system calculations using Erlang B or C based on traffic model selection
    """
    try:
        # Extract and convert form data to float
        time_slots = float(form_data['times_slots_per_carrier'])
        area = float(form_data['area'])
        users = float(form_data['number_of_users'])
        calls_per_day = float(form_data['average_calls_per_day'])
        call_duration = float(form_data['average_call_duration'])
        drop_prob = float(form_data['call_drop_probability'])
        traffic_model = form_data['traffic_model']  # New field
        min_sir = float(form_data['minimum_sir_required'])
        ref_power = float(form_data['measured_power_at_reference_distance'])
        ref_distance = float(form_data['reference_distance'])
        path_loss_exp = float(form_data['path_loss_exponent'])
        sensitivity = float(form_data['receiver_sensitivity'])
        
        results = {}
       
        
        # 1. Traffic calculations
        calls_per_hour = calls_per_day / 24.0
        traffic_per_user = calls_per_hour * call_duration / 60.0
        total_traffic = traffic_per_user * users
        
        results['Traffic per User (Erlangs)'] = f"{traffic_per_user:.4f}"
        results['Total Traffic (Erlangs)'] = f"{total_traffic:.2f}"
        
        # 2. Coverage calculations
        ref_power_watt = convertToLinear(ref_power)
        sensitivity_watt = sensitivity * 1e-6  # Convert μW to W
        max_distance = ref_distance / ((sensitivity_watt / ref_power_watt) ** (1.0 / path_loss_exp))
        
        results['Max Cell Radius (km)'] = f"{max_distance / 1000.0:.2f}"
        
        # 3. Cell planning (hexagonal cells)
        cell_area = 3.0 * math.sqrt(3.0) * (max_distance ** 2) / 2.0  # Hexagonal area
        cell_area_km2 = cell_area / 1e6  # Convert to km²
        cells_needed = math.ceil(area / cell_area_km2)
        
        results['Cell Area (km²)'] = f"{cell_area_km2:.2f}"
        results['Cells Required'] = f"{cells_needed}"
        
        # 4. Traffic per cell
        traffic_per_cell = total_traffic / cells_needed
        results['Traffic per Cell (Erlangs)'] = f"{traffic_per_cell:.2f}"
        
        # 5. Channel calculation using selected traffic model
        blocking_prob_percent = drop_prob * 100  # Convert to percentage
        
        if traffic_model == 'erlang_b':
            # Use Erlang B (Drop and Clear)
            channels_needed = find_channels_from_erlang_b(traffic_per_cell, blocking_prob_percent)
            
            if channels_needed is None:
                channels_needed = math.ceil(traffic_per_cell * 1.2)  # 20% margin
                calculation_method = "Approximation (Erlang B table not available)"
            else:
                calculation_method = f"Erlang B table (blocking prob: {blocking_prob_percent}%)"
                
        elif traffic_model == 'erlang_c':
            # Use Erlang C (Drop and Queue)
            channels_needed = erlang_c_channels(traffic_per_cell, drop_prob)
            
            if channels_needed is None:
                channels_needed = math.ceil(traffic_per_cell * 1.3)  # 30% margin for queuing
                calculation_method = "Approximation (Erlang C calculation failed)"
            else:
                calculation_method = f"Erlang C calculation (delay prob: {blocking_prob_percent}%)"
        
        else:
            # Fallback
            channels_needed = math.ceil(traffic_per_cell * 1.2)
            calculation_method = "Default approximation"
        
        carriers_needed = math.ceil(channels_needed / time_slots)
        
        results['Traffic Model Used'] = f"{traffic_model.replace('_', ' ').title()}"
        results['Channel Calculation Method'] = calculation_method
        results['Channels Needed per Cell'] = f"{channels_needed}"
        results['Carriers Required per Cell'] = f"{carriers_needed}"
        
        # 6. Frequency reuse calculation
        sir_linear = convertToLinear(min_sir)
        cluster_size = ((sir_linear * 6) ** (2 / path_loss_exp)) / 3
        cluster_size_rounded = max(1, round(cluster_size))

        # Ensure cluster size follows valid pattern
        valid_cluster_sizes = [1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 25, 27, 28, 31, 36, 37, 39, 43, 48, 49]
        cluster_size_final = next((x for x in valid_cluster_sizes if x >= cluster_size_rounded), valid_cluster_sizes[-1])
        
        results['Calculated Cluster Size'] = f"{cluster_size:.2f}"
        results['Cluster Size (N)'] = f"{cluster_size_final}"
        results['Frequency Reuse Factor'] = f"1/{cluster_size_final}"
        
        # 7. System capacity
        total_channels = carriers_needed * time_slots * cells_needed # we took 16 channel instead of 14 to account for extra 2 channels in each cell
        effective_channels = total_channels / cluster_size_final # wa accounted for that channels will be reused in system 
        
        results['Total Channels in System'] = f"{int(total_channels)}"
        results['Effective Channels'] = f"{int(effective_channels)}"
        
        
        # 9. Spectral efficiency
        if area > 0:
            spectral_eff = effective_channels / area
            results['Spectral Efficiency (channels/km²)'] = f"{spectral_eff:.2f}"
        
        
        # 11. System efficiency metrics
        channels_per_user = effective_channels / users if users > 0 else 0
        results['Channels per User'] = f"{channels_per_user:.4f}"
        
        coverage_efficiency = (cells_needed * cell_area_km2) / area
        results['Coverage Efficiency'] = f"{coverage_efficiency:.2%}"
        
        # 12. Traffic model specific information
        if traffic_model == 'erlang_b':
            results['Model Description'] = "Drop and Clear: Blocked calls are immediately dropped"
        else:
            results['Model Description'] = "Drop and Queue: Blocked calls wait in queue for available channels"

        try:
            llm2_client = GroqAPIClient()
            prompt1 = llm2_client.forge_prompt(form_data, results, analysis=AnalysisType2.CELLULAR)
            llm2_resp = llm2_client.generate_response(prompt1)
            results['llm_analysis2'] = markdown.markdown(llm2_resp)
        except Exception as e:
            print(f"Groq Error: {e}")
            results['llm_analysis2'] = "AI analysis temporarily unavailable."

        # GEMINI LLM analysis
        try:
            llm_client = GeminiAPIClient()
            prompt2 = llm_client.forge_prompt(form_data, results, analysis=AnalysisType.CELLULAR)
            llm_resp = llm_client.generate_response(prompt2)
            results['llm_analysis'] = markdown.markdown(llm_resp)
        except Exception as e:
            print(f"Gemini Error: {e}")
            results['llm_analysis'] = "AI analysis temporarily unavailable."

        return results
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Calculation error details: {error_details}")
        return {"Error": f"Calculation failed: {str(e)}", "Details": error_details}



def get_modulation_name(modulation_value):
    """Get modulation name from value"""
    modulations = {
        '2': 'BPSK',
        '4': 'QPSK', 
        '16': '16-QAM',
        '64': '64-QAM',
        '256': '256-QAM',
        '1024': '1024-QAM'
    }
    return modulations.get(str(modulation_value), 'Unknown')

def format_data_rate(rate_bps):
    """Format data rate with appropriate units"""
    if rate_bps >= 1e9:
        return f"{rate_bps/1e9:.2f} Gbps"
    elif rate_bps >= 1e6:
        return f"{rate_bps/1e6:.2f} Mbps"
    elif rate_bps >= 1e3:
        return f"{rate_bps/1e3:.2f} kbps"
    else:
        return f"{rate_bps:.0f} bps"

def calculate_ofdm_parameters(form_data):
    """
    Calculate OFDM system parameters including all data rates and spectral efficiency
    """
    try:
       
        # Extract and validate form data
        bandwidth = float(form_data['bandwidth'])  # kHz
        subcarrier_spacing = float(form_data['subcarrier_spacing'])  # kHz
        ofdm_symbols = int(form_data['ofdm_symbols'])
        modulation_type = int(form_data['modulation_type'])
        block_duration = float(form_data['block_duration'])  # ms
        resource_blocks = int(form_data['resource_blocks'])
        
        # Basic validation
        if bandwidth <= 0 or subcarrier_spacing <= 0:
            return {"Error": "Bandwidth and subcarrier spacing must be positive"}
        
        if subcarrier_spacing > bandwidth:
            return {"Error": "Subcarrier spacing cannot be larger than bandwidth"}
        
        if block_duration <= 0:
            return {"Error": "Block duration must be positive"}
        
        results = {}
        
        # Convert time units
        block_duration_seconds = block_duration / 1000  # convert ms to seconds
        symbol_duration_seconds = block_duration_seconds / ofdm_symbols  # duration per OFDM symbol
        
        # 1. Basic OFDM parameters
        bits_per_subcarrier = math.log2(modulation_type)
        number_of_subcarriers = math.floor(bandwidth / subcarrier_spacing)
        
       
        
        results['Modulation Scheme'] = get_modulation_name(modulation_type)
        results['Bits per Subcarrier'] = f"{bits_per_subcarrier:.0f}"
        results['Number of Subcarriers'] = f"{number_of_subcarriers:,}"
        results['Symbol Duration'] = f"{symbol_duration_seconds*1000:.3f} ms"
        
        # 2. Resource Elements calculations
        resource_elements_per_symbol = number_of_subcarriers
        resource_elements_per_block = resource_elements_per_symbol * ofdm_symbols
        total_resource_elements = resource_elements_per_block * resource_blocks
        
        results['Resource Elements per Symbol'] = f"{resource_elements_per_symbol:,}"
        results['Resource Elements per Block'] = f"{resource_elements_per_block:,}"
        results['Total Resource Elements'] = f"{total_resource_elements:,}"
        
        # 3. Bits capacity calculations
        bits_per_resource_element = bits_per_subcarrier
        bits_per_ofdm_symbol = bits_per_resource_element * resource_elements_per_symbol
        bits_per_resource_block = bits_per_ofdm_symbol * ofdm_symbols
        total_bits_per_frame = bits_per_resource_block * resource_blocks
        
        results['Bits per Resource Element'] = f"{bits_per_resource_element:.0f}"
        results['Bits per OFDM Symbol'] = f"{bits_per_ofdm_symbol:,}"
        results['Bits per Resource Block'] = f"{bits_per_resource_block:,}"
        results['Total Bits per Frame'] = f"{total_bits_per_frame:,}"
        
        # 4. DATA RATES - This is what was missing!
        
        # 4.1 Data rate for Resource Elements
        resource_element_rate_bps = bits_per_resource_element / symbol_duration_seconds
        results['Resource Element Data Rate'] = format_data_rate(resource_element_rate_bps)
        
        # 4.2 Data rate for OFDM Symbols  
        ofdm_symbol_rate_bps = bits_per_ofdm_symbol / symbol_duration_seconds
        results['OFDM Symbol Data Rate'] = format_data_rate(ofdm_symbol_rate_bps)
        
        # 4.3 Data rate for Resource Blocks
        resource_block_rate_bps = bits_per_resource_block / block_duration_seconds
        results['Resource Block Data Rate'] = format_data_rate(resource_block_rate_bps)
        
        # 4.4 Maximum transmission capacity using parallel resource blocks
        max_transmission_rate_bps = total_bits_per_frame / block_duration_seconds
        results['Maximum Transmission Capacity/rate'] = format_data_rate(max_transmission_rate_bps)
        
        # 5. Spectral Efficiency
        bandwidth_hz = bandwidth * 1000  # convert kHz to Hz
        spectral_efficiency = max_transmission_rate_bps / bandwidth_hz  # bits/s/Hz
        spectral_efficiency_per_rb = resource_block_rate_bps / bandwidth_hz  # per resource block
        
        results['Spectral Efficiency (Total)'] = f"{spectral_efficiency:.3f} bits/s/Hz"
        results['Spectral Efficiency per RB'] = f"{spectral_efficiency_per_rb:.3f} bits/s/Hz"
        
        # 6. Symbol and subcarrier rates
        symbol_rate_total = number_of_subcarriers / symbol_duration_seconds  # total symbols per second
        subcarrier_symbol_rate = 1 / symbol_duration_seconds  # symbols per second per subcarrier
        
        results['Total Symbol Rate'] = f"{symbol_rate_total/1000:.2f} kSymbols/s"
        results['Subcarrier Symbol Rate'] = f"{subcarrier_symbol_rate:.0f} Symbols/s"
        
        
        # 8. System parameters summary
        results['Resource Blocks'] = f"{resource_blocks}"
        results['OFDM Symbols per Block'] = f"{ofdm_symbols}"
        results['Subcarrier Spacing'] = f"{subcarrier_spacing} kHz"
        results['Total Bandwidth'] = f"{bandwidth} kHz"
        
        # Protected LLM call with rate limiting
        try:
            llm2_client = GroqAPIClient()
            prompt1 = llm2_client.forge_prompt(form_data, results, analysis=AnalysisType2.OFDM)
            llm2_resp = llm2_client.generate_response(prompt1)
            results['llm_analysis2'] = markdown.markdown(llm2_resp)
        except Exception as e:
            print(f"Groq Error: {e}")
            results['llm_analysis2'] = "AI analysis temporarily unavailable."

        # GEMINI LLM analysis
        try:
            llm_client = GeminiAPIClient()
            prompt2 = llm_client.forge_prompt(form_data, results, analysis=AnalysisType.OFDM)
            llm_resp = llm_client.generate_response(prompt2)
            results['llm_analysis'] = markdown.markdown(llm_resp)
        except Exception as e:
            print(f"Gemini Error: {e}")
            results['llm_analysis'] = "AI analysis temporarily unavailable."
        
      
        return results
        
    except ValueError as ve:
        error_msg = f"Invalid input value: {str(ve)}"
        print(f"ValueError: {error_msg}")
        return {"Error": error_msg}
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_msg = f"Calculation failed: {str(e)}"
        print(f"Exception: {error_msg}")
        print(f"Details: {error_details}")
        return {"Error": error_msg}
    


EB_N0_VALUES = {
        "BPSK": {
        '10^-1': 0.86,
        '10^-1.5': 2.37,
        '10^-2': 4.32,
        '10^-2.5': 5.71,
        '10^-3': 6.79,
        '10^-3.5': 7.66,
        '10^-4': 8.40,
        '10^-4.5': 9.03,
        '10^-5': 9.59,
        '10^-5.5': 10.08,
        '10^-6': 10.53,
        '10^-6.5': 10.94,
        '10^-7': 11.31,
        '10^-7.5': 11.65,
        '10^-8': 11.97
        },
        'QPSK': {
        '10^-1': 0.86,
        '10^-1.5': 2.37,
        '10^-2': 4.32,
        '10^-2.5': 5.71,
        '10^-3': 6.79,
        '10^-3.5': 7.66,
        '10^-4': 8.40,
        '10^-4.5': 9.03,
        '10^-5': 9.59,
        '10^-5.5': 10.08,
        '10^-6': 10.53,
        '10^-6.5': 10.94,
        '10^-7': 11.31,
        '10^-7.5': 11.65,
        '10^-8': 11.97
        },        
        '8-PSK': {
        '10^-1': 1.03,
        '10^-1.5': 4.37,
        '10^-2': 6.51,
        '10^-2.5': 8.18,
        '10^-3': 9.81,
        '10^-3.5': 10.98,
        '10^-4': 11.92,
        '10^-4.5': 12.21,
        '10^-5': 13.08,
        '10^-5.5': 13.34,
        '10^-6': 14.02,
        '10^-6.5': 14.31,
        '10^-7': 14.84,
        '10^-7.5': 15.04,
        '10^-8': 15.50
        },
        '16-PSK': {
        '10^-1': 4.01,
        '10^-1.5': 9.37,
        '10^-2': 11.11,
        '10^-2.5': 12.91,
        '10^-3': 14.32,
        '10^-3.5': 15.63,
        '10^-4': 16.07,
        '10^-4.5': 16.89,
        '10^-5': 17.37,
        '10^-5.5': 18.02,
        '10^-6': 18.54,
        '10^-6.5': 18.99,
        '10^-7': 19.38,
        '10^-7.5': 19.73,
        '10^-8': 20.03
    }
}

if __name__ == '__main__':  # Fixed: was '__main__' with wrong quotes
    app.run(host='0.0.0.0', debug=True)

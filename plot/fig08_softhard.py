import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def plot_calibrated_comparison():
    """
    Finds an optimal scaling factor 'n' for the calculated data to best fit
    the experimental data by minimizing the RMS error over the first 12 experimental points,
    then plots the calibrated comparison.
    """
    # 기본 경로 설정
    base_path = "/nas_2/transcendence/_delete/cowork/my_work/code/oldset/Exp_valider"
    exp_data_base_path = "/nas_2/transcendence/_delete/cowork/my_work/code/E_algo_final/xi_fit/hist"
    calc_data_base_path = os.path.join(base_path, "math")
    output_base_path = "/nas_2/transcendence/revision/exports/submission_package/main/Images/softhard"
    mapping_file_path = os.path.join(base_path, "comparer.txt")

    # 파라미터 정의
    anions = ["fsi"]
    temperatures = ["298", "353", "373", "423"]
    softhards = ["soft", "hard"]
    colors = ["#5555FF", "#55FF55", "#FFAA55", "#FF5555"]

    # 1. 매핑 파일 읽기
    mapper = {}
    try:
        with open(mapping_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    anion, temp, weird_num = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    mapper[(anion, temp)] = weird_num
    except FileNotFoundError:
        print(f"오류: 매핑 파일을 찾을 수 없습니다. 경로: {mapping_file_path}")
        return

    # 2. 모든 조합에 대해 반복하며 보정 및 플롯 생성
    for sh in softhards:
        for anion in anions:
            plt.figure(figsize=(10, 6))
            
            for i, temp in enumerate(temperatures):
                print(f"처리 중: {sh}, {anion}, {temp}K")

                weird_num = mapper.get((anion, temp))
                if not weird_num:
                    print(f"  - 경고: {anion}, {temp}에 대한 매핑을 찾을 수 없어 건너뜁니다.")
                    continue

                # --- 파일 경로 구성 ---
                exp_file = os.path.join(exp_data_base_path, sh, anion, f"exp_1.0_{temp}.txt")
                calc_file = os.path.join(calc_data_base_path, f"{weird_num}_{sh}.txt")
                
                try:
                    # --- 데이터 읽기 ---
                    exp_data = np.loadtxt(exp_file)
                    calc_y = np.loadtxt(calc_file)
                    calc_x = np.arange(1, len(calc_y) + 1)

                    if len(exp_data) < 2:
                        print(f"  - 경고: 실험 데이터가 너무 적어 건너뜁니다.")
                        continue
                    
                    mini = 0 if anion == "beti" and sh == "hard" and temp=="298" else 1
                    exp_x, exp_y = exp_data[:, 0], exp_data[:, 1]
                    maxi = min(13,len(exp_x))
                    real_maxi = maxi
                    for j in range(1,maxi):
                        if exp_x[j]>1000:
                            real_maxi = j-1
                            break
                    target_x, target_y = exp_x[mini:real_maxi],exp_y[mini:real_maxi]
                    interpolated_y = np.interp(target_x, calc_x, calc_y)

                    # 분자 계산: sum(a_i * b_i)
                    numerator = np.sum(interpolated_y * target_y)

                    # 분모 계산: sum(a_i^2)
                    denominator = np.sum(interpolated_y**2)
                    best_n = numerator / denominator

                    print(f"  - 최적 n 찾음: {best_n:.4f}")

                    # --- 보정된 데이터로 플로팅 ---
                    exp_color = colors[i]
                    
                    # Darken the color for the calculated line
                    rgb = mcolors.to_rgb(exp_color)
                    hsv = mcolors.rgb_to_hsv(rgb)
                    hsv[2] *= 0.8 # Slightly darken by reducing value/brightness
                    calc_color = mcolors.hsv_to_rgb(hsv)

                    plt.plot(exp_x, exp_y, label=f"MD {temp} K", linestyle='-', linewidth=2, zorder=5, color=exp_color, alpha=0.7)
                    
                    calibrated_calc_y = calc_y * best_n
                    
                    plot_calc_x = calc_x
                    plot_calibrated_calc_y = calibrated_calc_y

                    if sh == "hard":
                        # Find the peak of the calculated data
                        peak_calc_x_index = np.argmax(calibrated_calc_y)
                        peak_calc_x = calc_x[peak_calc_x_index]
                        
                        # Use the start of the experimental data for midpoint calculation
                        start_exp_x = exp_x[mini]
                        
                        # Calculate the midpoint x-value
                        midpoint_x = (peak_calc_x + start_exp_x) / 2
                        
                        # Find the closest index in calc_x to start plotting from
                        start_plot_index = np.abs(calc_x - midpoint_x).argmin()
                        
                        # Slice the data for plotting
                        plot_calc_x = calc_x[start_plot_index:]
                        plot_calibrated_calc_y = calibrated_calc_y[start_plot_index:]

                    plt.plot(plot_calc_x, plot_calibrated_calc_y, label=f"Prediction {temp} K", linestyle='dotted', linewidth=1.5, color=calc_color, zorder=10)
                except FileNotFoundError as e:
                    print(f"  - 경고: 데이터 파일을 찾을 수 없어 건너뜁니다. 상세: {e}")
                except Exception as e:
                    print(f"  - 오류: 예상치 못한 오류가 발생했습니다. 상세: {e}")

            # --- 플롯 속성 설정 ---
            plt.yscale('log')
            plt.ylabel(f'$h_{{{sh}}}(n)$')
            plt.xlabel('n(ps)')
            plt.ylim(1e-4, 1e0)
            
            xlim_map = {
                ("hard", "beti"): 5000,
                ("hard", "fsi"): 1000,
                ("hard", "tfsi"): 5000,
                ("soft", "beti"): 1000,
                ("soft", "fsi"): 400,
                ("soft", "tfsi"): 1000,
            }
            plt.xlim(0, xlim_map.get((sh, anion), 1000))

            # 텍스트 스타일 설정
            text_properties = dict(
                transform=plt.gca().transAxes, # 축 기준 좌표계 사용
                fontsize=14, 
                #fontweight='bold', 
                verticalalignment='top', # 텍스트의 상단을 y 위치에 맞춤
                horizontalalignment='left' # 텍스트의 좌단을 x 위치에 맞춤
            )
            plt.text(0.02, 0.98, f'{anion.upper()}, {sh.capitalize()}', **text_properties)

            if sh == "soft":
                plt.legend()
            plt.grid(True, which="both", ls="--", linewidth=0.5)
            
            output_file = os.path.join(output_base_path, f"{sh}_{anion}.pdf")
            plt.savefig(output_file)
            plt.close()
            print(f"  - 플롯 저장 완료: {output_file}")

    print("\n모든 보정 및 플로팅 작업을 완료했습니다.")

if __name__ == "__main__":
    plot_calibrated_comparison()
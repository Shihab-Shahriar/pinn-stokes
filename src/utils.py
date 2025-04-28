import os
import csv

def save_parameters_to_csv(model, output_dir="model_params"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over model parameters
    for name, param in model.named_parameters():
        param_data = param.detach().cpu().numpy()  # Convert the tensor to numpy array

        # Save the parameter data to CSV file
        csv_filename = os.path.join(output_dir, f"{name}.csv")
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # If the parameter is 2D, write rows separately, otherwise just write the flat vector
            if param_data.ndim == 2:
                writer.writerows(param_data)
            else:
                writer.writerow(param_data)

    print(f"Parameters saved to {output_dir} directory.")


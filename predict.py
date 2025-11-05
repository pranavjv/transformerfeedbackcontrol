def predict(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_predictions = []

    with torch.no_grad():  # No need to compute gradients during evaluation
        for batch in test_loader:
            rho, measurement_record, optimal_delta = [b.to(device) for b in batch]
            zeros = torch.zeros(optimal_delta.shape[0], 1).to(device)  # Create a tensor of zeros
            decoder_input = torch.cat((zeros, optimal_delta[:, :-1]), dim=1)  # Append the tensor of zeros at the beginning
            outputs = model(rho, decoder_input)
            all_predictions.append(outputs.cpu().numpy())

    return np.concatenate(all_predictions, axis=0)  # Convert the list of predictions to a numpy array

# Usage:
predictions = predict(model, test_dataloader)
print(predictions)


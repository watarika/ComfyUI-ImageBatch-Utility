class ImageBatchToSubBatchesNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "batch_size": ("INT", { "default": 1, "min": 1, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    CATEGORY = "image"

    def execute(self, image, batch_size):
        """
        Splits the image batch into smaller batches of the specified size.
        
        Parameters:
        image (torch.Tensor): A batch of images in the form of a PyTorch tensor.
        batch_size (int): The number of images in each smaller batch.
        
        Returns:
        tuple: A tuple containing a list of smaller batches.
        """
        num_batches = (image.shape[0] + batch_size - 1) // batch_size  # Calculate the number of batches
        return ([image[i*batch_size:(i+1)*batch_size] for i in range(num_batches)], )


NODE_CLASS_MAPPINGS = {
    "ImageBatchToSubBatches": ImageBatchToSubBatchesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchToSubBatches": "Image Batch To Sub Batches",
}

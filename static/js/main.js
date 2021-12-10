$(document).ready(function () {
    // Init
    $('#result').hide();
    $('.image-section').hide();
    $('.loader').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').attr("src" , e.target.result)
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                var data = data.split("-")
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#Rimg').attr("src",data[0]);
                $('#object').text(data[1])
                $('#object').show()
                console.log('Success!');
                console.log(data)
            },
        });
    });

});

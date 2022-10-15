function clicked_submit() {
    console.log(document.getElementById("form").Make.value)
    const Make = document.getElementById("form").Make.value
    const Type = document.getElementById("form").Type.value
    const Year = document.getElementById("form").Year.value
    const Origin = document.getElementById("form").Origin.value
    const Color = document.getElementById("form").Color.value
    const Options = document.getElementById("form").Options.value
    const Engine_Size = document.getElementById("form").Engine_Size.value
    const Fuel_Type = document.getElementById("form").Fuel_Type.value
    const Gear_Type = document.getElementById("form").Gear_Type.value
    const Mileage = document.getElementById("form").Mileage.value

    const url = "http://localhost:3000/?Make="+Make+"&Type="+Type+"&Year="+Year+"&Origin="+Origin+"&Color="+Color+"&Options="+Options+"&Engine_Size="+Engine_Size+"&Fuel_Type="+Fuel_Type+"&Gear_Type="+Gear_Type+"&Mileage="+Mileage//+"&Region=Riyadh"

    $("#result").html("Predicting....");
    $.get( url, function( data ) {
        pred = data[0].replace("[", "")
        $( "#result" ).html("SAR "+pred.replace("]", ""));
        console.log(data)
        console.log(url)

      });
}

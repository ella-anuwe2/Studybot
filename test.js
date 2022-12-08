var car1 = {
    name: "BMW",
    model: "i8",
    display: function(){
        return this.name + this.model
    }
}

var newCar = {
    name:"Audi",
    model:"A8"
}

document.write(car1.display.call(newCar))
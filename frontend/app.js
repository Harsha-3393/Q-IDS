async function predict(){

try{

document
.getElementById(
"result"
)
.innerHTML=
"Detecting...";

const data={

dur:0.12,
state:1,
dpkts:8,
sbytes:4500,
dbytes:2100,
rate:65,
sttl:252,
dttl:252,
sload:300000,
dload:250000,
sinpkt:5,
dinpkt:6,
sjit:0.3,
djit:0.2,
tcprtt:0.05,
synack:0.02,
ackdat:0.01,
smean:200,
dmean:150,
ct_state_ttl:2

};

const response=
await fetch(
"http://127.0.0.1:8000/predict",
{
method:"POST",

headers:{
"Content-Type":
"application/json"
},

body:
JSON.stringify(
data
)

}

);

const result=
await response.json();

document
.getElementById(
"result"
)
.innerHTML=

result
.prediction
.toUpperCase();

const history=
document
.getElementById(
"history"
);

// Remove placeholder
if(
history.innerHTML.includes(
"No predictions yet"
)
){
history.innerHTML="";
}

const row=
document
.createElement(
"li"
);

row.innerText=

new Date()
.toLocaleTimeString()

+

" - "

+

result.prediction;

history.prepend(
row
);

}

catch(err){

document
.getElementById(
"result"
)
.innerHTML=
"API ERROR";

console.log(err);

}

}

document
.querySelector(
"button"
)
.addEventListener(
"click",
predict);
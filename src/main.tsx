import { createRoot } from 'react-dom/client'
import './style.css'

import App from './App'
//import { StrictMode } from "react"
import React from 'react'
import { StrictMode } from 'react'
import main from './main.ts'

createRoot(document.getElementById('root')!).render(
    <StrictMode>
        <h1>bzz</h1>
        <App />
    </StrictMode>
)

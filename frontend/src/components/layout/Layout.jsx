import React from 'react';
import Sidebar from './Sidebar';
import Header from './Header';
import MobileNav from './MobileNav';

import { useApp } from '../../context/AppContext';

function Layout({ children }) {
  const { notification } = useApp();

  return (
    <div className="flex h-screen bg-slate-50 overflow-hidden">
      <Sidebar />
      
      <div className="flex-1 flex flex-col min-w-0">
       
        <Header />
        
        <main className="flex-1 overflow-y-auto">
          <div className="container mx-auto px-4 py-6 pb-24 md:pb-8 max-w-7xl">
            {children}
          </div>
        </main>
        
        
        <MobileNav />
      </div>

    </div>
  );
}

export default Layout;